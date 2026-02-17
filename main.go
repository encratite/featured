package main

import (
	"fmt"
	"io"
	"os"
	"time"

	"github.com/cdipaolo/goml/linear"
	"github.com/encratite/commons"
	"github.com/encratite/ohlc"
	"github.com/olekukonko/tablewriter"
	"github.com/olekukonko/tablewriter/tw"
)

const (
	sessionEnd = time.Duration(21) * time.Hour
	daysPerWeek = 7
	bitcoinSymbol = "BTCUSDT"

	logisticMethod = "Batch Gradient Ascent"
	alpha = 0.001
	regularization = 0
	maxIterations = 10000

	enableMomentum = true
	enableReference = true
	enableIndex = true
	enableWeekdays = true

	featureHighlightThreshold = 0.0500
)

var configuration *Configuration

type Configuration struct {
	BinanceDirectory string `yaml:"binanceDirectory"`
	BarchartDirectory string `yaml:"barchartDirectory"`
	StartDate commons.SerializableDate `yaml:"startDate"`
	SplitDate commons.SerializableDate `yaml:"splitDate"`
	EndDate commons.SerializableDate `yaml:"endDate"`
	IndexSymbol string `yaml:"indexSymbol"`
	Assets []Asset `yaml:"assets"`
}

type Asset struct {
	Symbol string `yaml:"symbol"`
	StartDate *commons.SerializableDate `yaml:"startDate"`
}

type timePriceMap map[time.Time]float64

func main() {
	configuration = commons.LoadConfiguration[Configuration]("yaml/featured.yaml")
	analyzeData()
}

func analyzeData() {
	referenceMap := loadDailyRecords(bitcoinSymbol, nil)
	indexRecords := ohlc.MustReadBarchart(configuration.IndexSymbol, configuration.BarchartDirectory, ohlc.TimeFrameD1)
	header := []string{
		"Symbol",
	}
	if enableMomentum {
		header = append(header, "Momentum")
	}
	if enableReference {
		header = append(header, "BTC")
	}
	if enableIndex {
		header = append(header, configuration.IndexSymbol)
	}
	if enableWeekdays {
		weekdays := []string{
			// "Monday",
			"Tuesday",
			"Wednesday",
			"Thursday",
			"Friday",
			"Saturday",
		}
		header = append(header, weekdays...)
	}
	header = append(header, []string{
		"Intercept",
		"IS R²",
		"OOS R²",
	}...)
	rows := commons.ParallelMap(configuration.Assets, func (a Asset) []string {
		return getRegressionCells(a.Symbol, a.StartDate, referenceMap, indexRecords)
	})
	alignments := []tw.Align{
		tw.AlignDefault,
	}
	for len(alignments) < len(header) {
		alignments = append(alignments, tw.AlignRight)
	}
	tableConfig := tablewriter.WithConfig(tablewriter.Config{
		Header: tw.CellConfig{
			Formatting: tw.CellFormatting{AutoFormat: tw.Off},
			Alignment: tw.CellAlignment{Global: tw.AlignLeft},
		}},
	)
	alignmentConfig := tablewriter.WithAlignment(alignments)
	fmt.Printf("\n")
	table := tablewriter.NewTable(os.Stdout, tableConfig, alignmentConfig)
	table.Header(header)
	table.Bulk(rows)
	table.Render()
	fmt.Printf("\n")
	fmt.Printf("IS time range: from %s to %s\n", commons.GetDateString(configuration.StartDate.Time), commons.GetDateString(configuration.SplitDate.Time))
	fmt.Printf("OOS time range: from %s to %s\n", commons.GetDateString(configuration.SplitDate.Time), commons.GetDateString(configuration.EndDate.Time))
	fmt.Printf("\n")
}

func loadDailyRecords(symbol string, startDate *commons.SerializableDate) timePriceMap {
	records := ohlc.MustReadBinance(symbol, configuration.BinanceDirectory, ohlc.TimeFrameH1)
	output := timePriceMap{}
	for _, record := range records {
		if startDate != nil && record.Timestamp.Before(startDate.Time) {
			continue
		}
		timeOfDay := commons.GetTimeOfDay(record.Timestamp)
		if timeOfDay == sessionEnd {
			date := commons.GetDate(record.Timestamp)
			output[date] = record.Close
		}
	}
	return output
}

func getRegressionCells(symbol string, startDate *commons.SerializableDate, referenceMap timePriceMap, indexRecords []ohlc.Record) []string {
	assetMap := loadDailyRecords(symbol, startDate)
	trainingFeatures := [][]float64{}
	trainingLabels := []float64{}
	testFeatures := [][]float64{}
	testLabels := []float64{}
	for i := 1; i < len(indexRecords); i++ {
		currentIndexRecord := indexRecords[i]
		previousIndexRecord := indexRecords[i - 1]
		if currentIndexRecord.Timestamp.Before(configuration.StartDate.Time) || configuration.EndDate.Before(currentIndexRecord.Timestamp) {
			continue
		}
		currentAssetClose, exists := assetMap[currentIndexRecord.Timestamp]
		if !exists {
			continue
		}
		previousAssetClose, exists := assetMap[previousIndexRecord.Timestamp]
		if !exists {
			continue
		}
		nextCloseTimestamp := currentIndexRecord.Timestamp.AddDate(0, 0, 1)
		nextAssetClose, exists := assetMap[nextCloseTimestamp]
		if !exists {
			continue
		}
		currentReferenceClose, exists := referenceMap[currentIndexRecord.Timestamp]
		if !exists {
			continue
		}
		previousReferenceClose, exists := referenceMap[previousIndexRecord.Timestamp]
		if !exists {
			continue
		}
		assetMomentum := getRateOfChange(currentAssetClose, previousAssetClose)
		var referenceMomentum float64
		if symbol != bitcoinSymbol {
			referenceMomentum = getRateOfChange(currentReferenceClose, previousReferenceClose)
		} else {
			referenceMomentum = 0.0
		}
		indexMomentum := getRateOfChange(currentIndexRecord.Close, previousIndexRecord.Close)
		dailyFeatures := []float64{}
		if enableMomentum {
			dailyFeatures = append(dailyFeatures, assetMomentum)
		}
		if enableReference {
			dailyFeatures = append(dailyFeatures, referenceMomentum)
		}
		if enableIndex {
			dailyFeatures = append(dailyFeatures, indexMomentum)
		}
		if enableWeekdays {
			weekday := nextCloseTimestamp.Weekday()
			weekdayIndex := (int(weekday) - 1) % daysPerWeek
			for j := 1; j < daysPerWeek - 1; j++ {
				var value float64
				if j == weekdayIndex {
					value = 1.0
				} else {
					value = 0.0
				}
				dailyFeatures = append(dailyFeatures, value)
			}
		}
		label := getRateOfChange(nextAssetClose, currentAssetClose)
		if currentIndexRecord.Timestamp.Before(configuration.SplitDate.Time) {
			trainingFeatures = append(trainingFeatures, dailyFeatures)
			trainingLabels = append(trainingLabels, label)
		} else {
			testFeatures = append(testFeatures, dailyFeatures)
			testLabels = append(testLabels, label)
		}
	}
	model := linear.NewLeastSquares(logisticMethod, alpha, regularization, maxIterations, trainingFeatures, trainingLabels)
	model.Output = io.Discard
	err := model.Learn()
	if err != nil {
		commons.Fatalf("Failed to fit model: %v", err)
	}
	cells := []string{
		commons.White(symbol),
	}
	addParameter := func (index int) {
		parameter := model.Parameters[index]
		var cell string
		if parameter != 0.0 {
			cell = fmt.Sprintf("%.4f", parameter)
		} else {
			cell = "-"
		}
		if parameter >= featureHighlightThreshold {
			cell = commons.Green(cell)
		} else if parameter <= - featureHighlightThreshold {
			cell = commons.Red(cell)
		}
		cells = append(cells, cell)
	}
	for j := 1; j < len(model.Parameters); j++ {
		addParameter(j)
	}
	addParameter(0)
	addR2Score := func (r2Score float64) {
		cell := commons.FormatPercentage(r2Score, 2)
		cells = append(cells, cell)
	}
	isR2Score := getR2Score(trainingFeatures, trainingLabels, model)
	addR2Score(isR2Score)
	oosR2Score := getR2Score(testFeatures, testLabels, model)
	addR2Score(oosR2Score)
	return cells
}

func getRateOfChange(a, b float64) float64 {
	return a / b - 1.0
}

func getR2Score(features [][]float64, labels []float64, model *linear.LeastSquares) float64 {
	meanObserved := commons.Mean(labels)
	residualSum := 0.0
	totalSum := 0.0
	for i, f := range features {
		label := labels[i]
		prediction, err := model.Predict(f)
		if err != nil {
			commons.Fatalf("Prediction failed: %v", err)
		}
		residualDelta := label - prediction[0]
		residualSum += residualDelta * residualDelta
		totalDelta := label - meanObserved
		totalSum += totalDelta * totalDelta
	}
	r2Score := 1.0 - residualSum / totalSum
	return r2Score
}