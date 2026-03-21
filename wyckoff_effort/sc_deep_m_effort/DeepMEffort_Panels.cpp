// ==========================================================================
// Deep-M Effort Panels — Companion Study for Subplots
// File: DeepMEffort_Panels.cpp
//
// Displays Effort Index, Volume Ratio, and Delta % as separate chart
// regions below the price chart. Reads computed data from the main
// DeepMEffort_NQ study via sc.GetStudyArrayUsingID().
//
// Setup:
//   1. Add DeepMEffort_NQ study first (it computes all metrics)
//   2. Add this study — it will auto-detect DeepMEffort_NQ
//   3. Each panel appears in its own chart region
//
// Compile: Analysis > Build Custom Studies DLL > Remote Build
// ==========================================================================

#include "sierrachart.h"
#include <cmath>

SCDLLName("DeepMEffort_Panels")

// =========================================================================
// Panel 1: Effort Index with EMA and Bollinger Bands
// =========================================================================
SCSFExport scsf_DeepMEffort_EffortPanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef EffortLine  = sc.Subgraph[0];
    SCSubgraphRef EffortEMA   = sc.Subgraph[1];
    SCSubgraphRef UpperBand   = sc.Subgraph[2];
    SCSubgraphRef LowerBand   = sc.Subgraph[3];

    SCInputRef InSourceStudy  = sc.Input[0];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Deep-M Effort Index";
        sc.AutoLoop = 1;
        sc.GraphRegion = 1;

        EffortLine.Name = "Effort Index";
        EffortLine.DrawStyle = DRAWSTYLE_LINE;
        EffortLine.PrimaryColor = RGB(33, 150, 243);  // Blue
        EffortLine.LineWidth = 2;

        EffortEMA.Name = "Effort EMA";
        EffortEMA.DrawStyle = DRAWSTYLE_LINE;
        EffortEMA.PrimaryColor = RGB(255, 152, 0);    // Orange
        EffortEMA.LineWidth = 1;

        UpperBand.Name = "Upper Band";
        UpperBand.DrawStyle = DRAWSTYLE_DASH;
        UpperBand.PrimaryColor = RGB(200, 200, 200);   // Gray
        UpperBand.LineWidth = 1;

        LowerBand.Name = "Lower Band";
        LowerBand.DrawStyle = DRAWSTYLE_DASH;
        LowerBand.PrimaryColor = RGB(200, 200, 200);
        LowerBand.LineWidth = 1;

        InSourceStudy.Name = "Source Study (DeepMEffort_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        return;
    }

    // Read from DeepMEffort_NQ study subgraphs:
    //   Subgraph[3] = Effort Index
    //   Subgraph[4] = Effort EMA
    //   Subgraph[5] = Effort Upper Band
    //   Subgraph[6] = Effort Lower Band
    int sourceStudyID = InSourceStudy.GetStudyID();

    SCFloatArray srcEffort;
    SCFloatArray srcEMA;
    SCFloatArray srcUpper;
    SCFloatArray srcLower;

    sc.GetStudyArrayUsingID(sourceStudyID, 3, srcEffort);
    sc.GetStudyArrayUsingID(sourceStudyID, 4, srcEMA);
    sc.GetStudyArrayUsingID(sourceStudyID, 5, srcUpper);
    sc.GetStudyArrayUsingID(sourceStudyID, 6, srcLower);

    EffortLine[sc.Index] = srcEffort[sc.Index];
    EffortEMA[sc.Index]  = srcEMA[sc.Index];
    UpperBand[sc.Index]  = srcUpper[sc.Index];
    LowerBand[sc.Index]  = srcLower[sc.Index];
}

// =========================================================================
// Panel 2: Volume Ratio (current volume / volume EMA)
// =========================================================================
SCSFExport scsf_DeepMEffort_VolumePanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef VolRatioLine = sc.Subgraph[0];
    SCSubgraphRef AbsThreshLine = sc.Subgraph[1];
    SCSubgraphRef VacThreshLine = sc.Subgraph[2];

    SCInputRef InSourceStudy   = sc.Input[0];
    SCInputRef InAbsThreshold  = sc.Input[1];
    SCInputRef InVacThreshold  = sc.Input[2];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Deep-M Volume Ratio";
        sc.AutoLoop = 1;
        sc.GraphRegion = 2;

        VolRatioLine.Name = "Vol Ratio";
        VolRatioLine.DrawStyle = DRAWSTYLE_BAR;
        VolRatioLine.PrimaryColor = RGB(100, 181, 246);  // Light blue
        VolRatioLine.SecondaryColor = RGB(239, 83, 80);   // Red for high
        VolRatioLine.SecondaryColorUsed = 1;
        VolRatioLine.LineWidth = 3;

        AbsThreshLine.Name = "Absorption Threshold";
        AbsThreshLine.DrawStyle = DRAWSTYLE_LINE;
        AbsThreshLine.PrimaryColor = RGB(76, 175, 80);
        AbsThreshLine.LineWidth = 1;

        VacThreshLine.Name = "Vacuum Threshold";
        VacThreshLine.DrawStyle = DRAWSTYLE_LINE;
        VacThreshLine.PrimaryColor = RGB(156, 39, 176);
        VacThreshLine.LineWidth = 1;

        InSourceStudy.Name = "Source Study (DeepMEffort_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        InAbsThreshold.Name = "Absorption Threshold";
        InAbsThreshold.SetFloat(1.8f);

        InVacThreshold.Name = "Vacuum Threshold";
        InVacThreshold.SetFloat(0.5f);

        return;
    }

    int sourceStudyID = InSourceStudy.GetStudyID();

    // Volume Ratio is stored in main study's Subgraph[0].Arrays[1]
    SCFloatArray srcVolRatio;
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, sourceStudyID, 0, 1, srcVolRatio);

    float vr = srcVolRatio[sc.Index];
    VolRatioLine[sc.Index] = vr;

    float absT = InAbsThreshold.GetFloat();
    float vacT = InVacThreshold.GetFloat();

    // Color bars: red when above absorption threshold
    if (vr >= absT)
        VolRatioLine.DataColor[sc.Index] = VolRatioLine.SecondaryColor;
    else
        VolRatioLine.DataColor[sc.Index] = VolRatioLine.PrimaryColor;

    AbsThreshLine[sc.Index] = absT;
    VacThreshLine[sc.Index] = vacT;
}

// =========================================================================
// Panel 3: Delta % and Absorption Score
// =========================================================================
SCSFExport scsf_DeepMEffort_DeltaPanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef DeltaPctLine   = sc.Subgraph[0];
    SCSubgraphRef DeltaEMALine   = sc.Subgraph[1];
    SCSubgraphRef AbsorptionLine = sc.Subgraph[2];
    SCSubgraphRef ZeroLine       = sc.Subgraph[3];

    SCInputRef InSourceStudy = sc.Input[0];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Deep-M Delta & Absorption";
        sc.AutoLoop = 1;
        sc.GraphRegion = 3;

        DeltaPctLine.Name = "Delta %";
        DeltaPctLine.DrawStyle = DRAWSTYLE_BAR;
        DeltaPctLine.PrimaryColor = RGB(76, 175, 80);    // Green (positive)
        DeltaPctLine.SecondaryColor = RGB(239, 83, 80);   // Red (negative)
        DeltaPctLine.SecondaryColorUsed = 1;
        DeltaPctLine.LineWidth = 3;

        DeltaEMALine.Name = "Delta % EMA";
        DeltaEMALine.DrawStyle = DRAWSTYLE_LINE;
        DeltaEMALine.PrimaryColor = RGB(255, 152, 0);    // Orange
        DeltaEMALine.LineWidth = 1;

        AbsorptionLine.Name = "Absorption Score";
        AbsorptionLine.DrawStyle = DRAWSTYLE_LINE;
        AbsorptionLine.PrimaryColor = RGB(255, 235, 59);  // Yellow
        AbsorptionLine.LineWidth = 2;

        ZeroLine.Name = "Zero";
        ZeroLine.DrawStyle = DRAWSTYLE_LINE;
        ZeroLine.PrimaryColor = RGB(100, 100, 100);
        ZeroLine.LineWidth = 1;

        InSourceStudy.Name = "Source Study (DeepMEffort_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        return;
    }

    int sourceStudyID = InSourceStudy.GetStudyID();

    // Delta Pct = main study Subgraph[0].Arrays[2]
    // Delta Pct EMA = main study Subgraph[0].Arrays[3]
    // Absorption Score = main study Subgraph[7]
    SCFloatArray srcDeltaPct;
    SCFloatArray srcDeltaEMA;
    SCFloatArray srcAbsorption;

    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, sourceStudyID, 0, 2, srcDeltaPct);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, sourceStudyID, 0, 3, srcDeltaEMA);
    sc.GetStudyArrayUsingID(sourceStudyID, 7, srcAbsorption);

    float dp = srcDeltaPct[sc.Index];
    DeltaPctLine[sc.Index] = dp;

    // Color delta bars: green positive, red negative
    if (dp >= 0)
        DeltaPctLine.DataColor[sc.Index] = DeltaPctLine.PrimaryColor;
    else
        DeltaPctLine.DataColor[sc.Index] = DeltaPctLine.SecondaryColor;

    DeltaEMALine[sc.Index]   = srcDeltaEMA[sc.Index];
    AbsorptionLine[sc.Index] = srcAbsorption[sc.Index];
    ZeroLine[sc.Index] = 0.0f;
}
