package main

import (
	"fmt"
	"io"
	"math"
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

	featureHighlightThreshold = 0.0500

	weeksPerYear = 52
)

var configuration *Configuration

type Configuration struct {
	BinanceDirectory string `yaml:"binanceDirectory"`
	BarchartDirectory string `yaml:"barchartDirectory"`
	StartDate commons.SerializableDate `yaml:"startDate"`
	SplitDate commons.SerializableDate `yaml:"splitDate"`
	EndDate commons.SerializableDate `yaml:"endDate"`
	IndexSymbol string `yaml:"indexSymbol"`
	EnableMomentum bool `yaml:"enableMomentum"`
	EnableReference bool `yaml:"enableReference"`
	EnableIndex bool `yaml:"enableIndex"`
	EnableWeekdays bool `yaml:"enableWeekdays"`
	EnableWeekdayFilter bool `yaml:"enableWeekdayFilter"`
	WeekdayFilter commons.SerializableWeekday `yaml:"weekdayFilter"`
	HoldingTime int `yaml:"holdingTime"`
	LongThreshold float64 `yaml:"longThreshold"`
	ShortThreshold float64 `yaml:"shortThreshold"`
	RiskFreeRate float64 `yaml:"riskFreeRate"`
	Assets []Asset `yaml:"assets"`
}

type Asset struct {
	Symbol string `yaml:"symbol"`
	StartDate *commons.SerializableDate `yaml:"startDate"`
}

type timePriceMap map[time.Time]float64

type regressionData struct {
	cells []string
	oosR2Score float64
}

func main() {
	configuration = commons.LoadConfiguration[Configuration]("yaml/featured.yaml")
	analyzeData()
}

func analyzeData() {
	referenceMap := loadDailyRecords(bitcoinSymbol, nil, true, false)
	indexMap := loadDailyRecords(configuration.IndexSymbol, nil, false, true)
	header := []string{
		"Symbol",
	}
	if configuration.EnableMomentum {
		header = append(header, "Momentum")
	}
	if configuration.EnableReference {
		header = append(header, "BTC")
	}
	if configuration.EnableIndex {
		header = append(header, configuration.IndexSymbol)
	}
	if configuration.EnableWeekdays {
		weekdays := []string{
			"Monday",
			"Tuesday",
			"Wednesday",
			"Thursday",
			"Friday",
			"Saturday",
			"Sunday",
		}
		header = append(header, weekdays...)
	}
	header = append(header, []string{
		"Intercept",
		"IS R²",
		"OOS R²",
		"Ret (Long)",
		"SR (Long)",
		"Ret (Short)",
		"SR (Short)",
	}...)
	data := commons.ParallelMap(configuration.Assets, func (a Asset) regressionData {
		return getRegressionCells(a.Symbol, a.StartDate, referenceMap, indexMap)
	})
	rows := [][]string{}
	oosR2Scores := []float64{}
	for _, d := range data {
		rows = append(rows, d.cells)
		oosR2Scores = append(oosR2Scores, d.oosR2Score)
	}
	medianR2Score := commons.Median(oosR2Scores)
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
	fmt.Printf("Median OOS R² score: %s\n", commons.FormatPercentage(medianR2Score, 2))
	if configuration.EnableWeekdayFilter {
		fmt.Printf("Weekday filter: %s\n", configuration.WeekdayFilter)
	}
	if configuration.HoldingTime > 1 {
		fmt.Printf("Holding time: %d days\n", configuration.HoldingTime)
	}
	fmt.Printf("\n")
}

func loadDailyRecords(symbol string, startDate *commons.SerializableDate, sessionEndFilter bool, barchart bool) timePriceMap {
	var records []ohlc.Record
	if barchart {
		records = ohlc.MustReadBarchart(symbol, configuration.BarchartDirectory, ohlc.TimeFrameD1)
	} else {
		records = ohlc.MustReadBinance(symbol, configuration.BinanceDirectory, ohlc.TimeFrameH1)
	}
	output := timePriceMap{}
	for _, record := range records {
		if startDate != nil && record.Timestamp.Before(startDate.Time) {
			continue
		}
		if sessionEndFilter {
			timeOfDay := commons.GetTimeOfDay(record.Timestamp)
			if timeOfDay == sessionEnd {
				date := commons.GetDate(record.Timestamp)
				output[date] = record.Close
			}
		} else {
			output[record.Timestamp] = record.Close
		}
	}
	return output
}

func getRegressionCells(symbol string, startDate *commons.SerializableDate, referenceMap timePriceMap, indexMap timePriceMap) regressionData {
	assetMap := loadDailyRecords(symbol, startDate, true, false)
	trainingFeatures := [][]float64{}
	trainingLabels := []float64{}
	testFeatures := [][]float64{}
	testLabels := []float64{}
	for date := configuration.StartDate.Time; date.Before(configuration.EndDate.Time); date = date.AddDate(0, 0, 1) {
		weekday := date.Weekday()
		if configuration.EnableWeekdayFilter && weekday != configuration.WeekdayFilter.Weekday {
			continue
		}
		currentIndexCloseDate, currentIndexClose, exists := getClosestRecord(date, indexMap)
		if !exists {
			continue
		}
		previousIndexCloseDate := currentIndexCloseDate.AddDate(0, 0, -1)
		_, previousIndexClose, exists := getClosestRecord(previousIndexCloseDate, indexMap)
		if !exists {
			continue
		}
		currentAssetClose, exists := assetMap[date]
		if !exists {
			continue
		}
		previousDate := date.AddDate(0, 0, -1)
		previousAssetClose, exists := assetMap[previousDate]
		if !exists {
			continue
		}
		nextCloseTimestamp := date.AddDate(0, 0, configuration.HoldingTime)
		nextAssetClose, exists := assetMap[nextCloseTimestamp]
		if !exists {
			continue
		}
		currentReferenceClose, exists := referenceMap[date]
		if !exists {
			continue
		}
		previousReferenceClose, exists := referenceMap[previousDate]
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
		indexMomentum := getRateOfChange(currentIndexClose, previousIndexClose)
		dailyFeatures := []float64{}
		if configuration.EnableMomentum {
			dailyFeatures = append(dailyFeatures, assetMomentum)
		}
		if configuration.EnableReference {
			dailyFeatures = append(dailyFeatures, referenceMomentum)
		}
		if configuration.EnableIndex {
			dailyFeatures = append(dailyFeatures, indexMomentum)
		}
		if configuration.EnableWeekdays {
			weekdayIndex := (int(weekday) + 6) % daysPerWeek
			for j := range daysPerWeek {
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
		if date.Before(configuration.SplitDate.Time) {
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
	longReturns, shortReturns := runBacktest(testFeatures, testLabels, model)
	addReturns := func (returns []float64) {
		totalReturn, sharpeRatio := analyzeReturns(returns)
		var totalReturnString, sharpeRatioString string
		if totalReturn != 0.0 {
			totalReturnString = commons.FormatPercentage(totalReturn, 2)
			sharpeRatioString = fmt.Sprintf("%.2f", sharpeRatio)
		} else {
			totalReturnString = "-"
			sharpeRatioString = "-"
		}
		cells = append(cells, []string{
			totalReturnString,
			sharpeRatioString,
		}...)
	}
	addReturns(longReturns)
	addReturns(shortReturns)
	data := regressionData{
		cells: cells,
		oosR2Score: oosR2Score,
	}
	return data
}

func getRateOfChange(a, b float64) float64 {
	return a / b - 1.0
}

func getR2Score(features [][]float64, labels []float64, model *linear.LeastSquares) float64 {
	meanObserved := commons.Mean(labels)
	residualSum := 0.0
	totalSum := 0.0
	for i := range features {
		label := labels[i]
		prediction, err := model.Predict(features[i])
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

func getClosestRecord(date time.Time, indexMap timePriceMap) (time.Time, float64, bool) {
	for range 10 {
		close, exists := indexMap[date]
		if exists {
			return date, close, true
		}
		date = date.AddDate(0, 0, -1)
	}
	return time.Time{}, math.NaN(), false
}

func runBacktest(features [][]float64, labels []float64, model *linear.LeastSquares) ([]float64, []float64) {
	longReturns := []float64{}
	shortReturns := []float64{}
	for i := range features {
		prediction, err := model.Predict(features[i])
		if err != nil {
			commons.Fatalf("Prediction failed: %v", err)
		}
		signal := prediction[0]
		label := labels[i]
		// fmt.Printf("signal = %.3f, label = %.3f\n", signal, label)
		if signal > configuration.LongThreshold {
			longReturns = append(longReturns, label)
		} else {
			longReturns = append(longReturns, 0.0)
		}
		if signal < configuration.ShortThreshold {
			shortReturn := 1.0 / (1.0 + label) - 1.0
			shortReturns = append(shortReturns, shortReturn)
		} else {
			shortReturns = append(shortReturns, 0.0)
		}
	}
	return longReturns, shortReturns
}

func analyzeReturns(returns []float64) (float64, float64) {
	totalReturn := 0.0
	for _, r := range returns {
		totalReturn += r
	}
	sharpeRatio := getSharpeRatio(returns)
	return totalReturn, sharpeRatio
}

func getSharpeRatio(weeklyReturns []float64) float64 {
	if len(weeklyReturns) < 2 {
		return math.NaN()
	}
	meanReturn := commons.Mean(weeklyReturns)
	stdDev := commons.StdDev(weeklyReturns)
	riskFreeRate := configuration.RiskFreeRate / weeksPerYear
	weeklySharpeRatio := (meanReturn - riskFreeRate) / stdDev
	sharpeRatio := math.Sqrt(weeksPerYear) * weeklySharpeRatio
	if math.IsInf(sharpeRatio, 1) || math.IsInf(sharpeRatio, -1) {
		return math.NaN()
	}
	return sharpeRatio
}