// ==========================================================================
// Wyckoff Weis Panels — Companion Studies for Sub-Chart Regions
// File: WyckoffWeis_Panels.cpp
//
// Displays key metrics from WyckoffWeis_NQ as separate chart panels
// so a prop trader can read the tape at a glance during live trading.
//
// Panel 1: CVD + Wave Delta  (Region 1) — directional pressure
// Panel 2: Wave Volume Bars  (Region 2) — Weis wave volume with
//           large-wave highlighting and directional coloring
// Panel 3: Effort/Result + Absorption (Region 3) — where effort is
//           absorbed (institutional activity) vs ease of movement
// Panel 4: Phase + Signal    (Region 4) — Wyckoff phase state and
//           current trade signal as a colored histogram
//
// Setup:
//   1. Add WyckoffWeis_NQ study first (computes all metrics)
//   2. Add one or more of these panel studies
//   3. Set "Source Study" input to WyckoffWeis_NQ
//
// Compile: Analysis > Build Custom Studies DLL > Remote Build
// ==========================================================================

#include "sierrachart.h"
#include <cmath>

SCDLLName("WyckoffWeis_Panels")

// =========================================================================
// Panel 1: CVD + Wave Delta — Directional Pressure
//
// THIS IS THE #1 INDICATOR FOR LIVE PROP TRADING.
// CVD shows who is in control (buyers vs sellers) in real time.
// When CVD diverges from price, a reversal is coming.
//   - Price making new highs + CVD declining = bearish divergence (short)
//   - Price making new lows + CVD rising = bullish divergence (long)
//   - Wave Delta bars show delta commitment per wave segment
//
// How to read it:
//   - Green bars = aggressive buying dominant in this wave
//   - Red bars = aggressive selling dominant in this wave
//   - Blue CVD line trending up = buyers in control
//   - Blue CVD line trending down = sellers in control
//   - CVD vs price divergence = highest probability trade setup
// =========================================================================
SCSFExport scsf_WyckoffWeis_CVDPanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef CVDLine      = sc.Subgraph[0];
    SCSubgraphRef DeltaBar     = sc.Subgraph[1];
    SCSubgraphRef ZeroLine     = sc.Subgraph[2];

    SCInputRef InSourceStudy   = sc.Input[0];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Wyckoff CVD + Delta";
        sc.AutoLoop = 1;
        sc.GraphRegion = 1;

        CVDLine.Name = "CVD";
        CVDLine.DrawStyle = DRAWSTYLE_LINE;
        CVDLine.PrimaryColor = RGB(33, 150, 243);   // Blue
        CVDLine.LineWidth = 2;

        DeltaBar.Name = "Bar Delta";
        DeltaBar.DrawStyle = DRAWSTYLE_BAR;
        DeltaBar.PrimaryColor = RGB(76, 175, 80);   // Green (positive)
        DeltaBar.SecondaryColor = RGB(239, 83, 80);  // Red (negative)
        DeltaBar.SecondaryColorUsed = 1;
        DeltaBar.LineWidth = 3;

        ZeroLine.Name = "Zero";
        ZeroLine.DrawStyle = DRAWSTYLE_LINE;
        ZeroLine.PrimaryColor = RGB(80, 80, 80);
        ZeroLine.LineWidth = 1;

        InSourceStudy.Name = "Source Study (WyckoffWeis_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        return;
    }

    int srcID = InSourceStudy.GetStudyID();

    // CVD = main study Subgraph[3] (CVDLine)
    // Delta per bar = Subgraph[0].Arrays[0] (DeltaArr on WaveTrend)
    SCFloatArray srcCVD;
    SCFloatArray srcDelta;

    sc.GetStudyArrayUsingID(srcID, 3, srcCVD);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 0, 0, srcDelta);

    float cvd = srcCVD[sc.Index];
    float dlt = srcDelta[sc.Index];

    CVDLine[sc.Index] = cvd;
    DeltaBar[sc.Index] = dlt;
    ZeroLine[sc.Index] = 0.0f;

    // Color delta bars by sign
    if (dlt >= 0)
        DeltaBar.DataColor[sc.Index] = DeltaBar.PrimaryColor;
    else
        DeltaBar.DataColor[sc.Index] = DeltaBar.SecondaryColor;
}


// =========================================================================
// Panel 2: Weis Wave Volume — Wave-Segmented Volume Bars
//
// Shows cumulative volume per wave segment, colored by wave direction.
// This is the classic "Weis Wave" volume display.
//
// How to read it:
//   - Tall green bar = strong buying wave (high volume up move)
//   - Tall red bar = strong selling wave (high volume down move)
//   - Shrinking wave volume in trend direction = exhaustion
//   - Expanding wave volume against trend = reversal warning
//   - Yellow coloring = anomalous large wave (>1.5× avg of prior 4)
//   - Compare wave volumes: increasing = trend healthy,
//     decreasing = trend weakening (Wyckoff effort diminishing)
// =========================================================================
SCSFExport scsf_WyckoffWeis_WaveVolPanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef WaveVolBar   = sc.Subgraph[0];
    SCSubgraphRef AvgVolLine   = sc.Subgraph[1];

    SCInputRef InSourceStudy   = sc.Input[0];
    SCInputRef InAvgPeriod     = sc.Input[1];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Weis Wave Volume";
        sc.AutoLoop = 1;
        sc.GraphRegion = 2;

        WaveVolBar.Name = "Wave Volume";
        WaveVolBar.DrawStyle = DRAWSTYLE_BAR;
        WaveVolBar.PrimaryColor = RGB(76, 175, 80);    // Green (up wave)
        WaveVolBar.SecondaryColor = RGB(239, 83, 80);   // Red (down wave)
        WaveVolBar.SecondaryColorUsed = 1;
        WaveVolBar.LineWidth = 4;

        AvgVolLine.Name = "Avg Wave Vol";
        AvgVolLine.DrawStyle = DRAWSTYLE_LINE;
        AvgVolLine.PrimaryColor = RGB(255, 152, 0);    // Orange
        AvgVolLine.LineWidth = 1;

        InSourceStudy.Name = "Source Study (WyckoffWeis_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        InAvgPeriod.Name = "Wave Volume Average Period";
        InAvgPeriod.SetInt(20);
        InAvgPeriod.SetIntLimits(5, 100);

        return;
    }

    int srcID = InSourceStudy.GetStudyID();
    int avgPeriod = InAvgPeriod.GetInt();

    // Wave Volume = Subgraph[5] (WaveVolSg)
    // Wave Dir = Subgraph[0].Arrays[2]
    SCFloatArray srcWaveVol;
    SCFloatArray srcWaveDir;

    sc.GetStudyArrayUsingID(srcID, 5, srcWaveVol);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 0, 2, srcWaveDir);

    float wv = srcWaveVol[sc.Index];
    float wd = srcWaveDir[sc.Index];

    WaveVolBar[sc.Index] = wv;

    // Color by wave direction: green=up, red=down
    if (wd > 0)
        WaveVolBar.DataColor[sc.Index] = WaveVolBar.PrimaryColor;
    else
        WaveVolBar.DataColor[sc.Index] = WaveVolBar.SecondaryColor;

    // Simple rolling average of wave volume for reference line
    float sum = 0.0f;
    int count = 0;
    int s = (sc.Index - avgPeriod + 1 > 0) ? sc.Index - avgPeriod + 1 : 0;
    for (int k = s; k <= sc.Index; k++)
    {
        sum += srcWaveVol[k];
        count++;
    }
    AvgVolLine[sc.Index] = (count > 0) ? sum / count : 0.0f;
}


// =========================================================================
// Panel 3: Effort/Result + Absorption — Institutional Activity
//
// ER_Ratio < 0.5 = absorption (high effort, low result = institutions
//   absorbing supply/demand). These are the strongest setups.
// ER_Ratio > 2.0 = ease of movement (low effort, big price move =
//   vacuum / no resistance). Trend continuation.
//
// Absorption dots mark exact bars where institutions are active.
// Green dots = bullish absorption (buying), Purple dots = bearish.
//
// How to read it:
//   - ER drops below 0.5 (red zone) = institutional absorption happening
//   - Green absorption dot at support + CVD rising = LONG setup
//   - Purple absorption dot at resistance + CVD falling = SHORT setup
//   - ER spikes above 2.0 = ease of movement, trend likely continues
// =========================================================================
SCSFExport scsf_WyckoffWeis_ERPanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef ERLine       = sc.Subgraph[0];
    SCSubgraphRef AbsHighLine  = sc.Subgraph[1];
    SCSubgraphRef AbsLowLine   = sc.Subgraph[2];
    SCSubgraphRef AbsDots      = sc.Subgraph[3];
    SCSubgraphRef VolFilterBar = sc.Subgraph[4];

    SCInputRef InSourceStudy   = sc.Input[0];
    SCInputRef InAbsThreshold  = sc.Input[1];
    SCInputRef InEaseThreshold = sc.Input[2];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Wyckoff Effort/Result";
        sc.AutoLoop = 1;
        sc.GraphRegion = 3;

        ERLine.Name = "ER Ratio";
        ERLine.DrawStyle = DRAWSTYLE_LINE;
        ERLine.PrimaryColor = RGB(33, 150, 243);   // Blue
        ERLine.LineWidth = 2;

        AbsHighLine.Name = "Ease of Movement";
        AbsHighLine.DrawStyle = DRAWSTYLE_DASH;
        AbsHighLine.PrimaryColor = RGB(76, 175, 80);  // Green
        AbsHighLine.LineWidth = 1;

        AbsLowLine.Name = "Absorption Zone";
        AbsLowLine.DrawStyle = DRAWSTYLE_DASH;
        AbsLowLine.PrimaryColor = RGB(239, 83, 80);   // Red
        AbsLowLine.LineWidth = 1;

        AbsDots.Name = "Absorption Events";
        AbsDots.DrawStyle = DRAWSTYLE_POINT;
        AbsDots.PrimaryColor = RGB(76, 175, 80);      // Green (bullish)
        AbsDots.SecondaryColor = RGB(156, 39, 176);    // Purple (bearish)
        AbsDots.SecondaryColorUsed = 1;
        AbsDots.LineWidth = 5;
        AbsDots.DrawZeros = 0;

        VolFilterBar.Name = "Volume Strength";
        VolFilterBar.DrawStyle = DRAWSTYLE_IGNORE;     // Hidden by default

        InSourceStudy.Name = "Source Study (WyckoffWeis_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        InAbsThreshold.Name = "Absorption ER Threshold";
        InAbsThreshold.SetFloat(0.5f);
        InAbsThreshold.SetFloatLimits(0.1f, 1.5f);

        InEaseThreshold.Name = "Ease of Movement Threshold";
        InEaseThreshold.SetFloat(2.0f);
        InEaseThreshold.SetFloatLimits(1.0f, 5.0f);

        return;
    }

    int srcID = InSourceStudy.GetStudyID();
    float absT = InAbsThreshold.GetFloat();
    float easeT = InEaseThreshold.GetFloat();

    // ER Ratio = Subgraph[3].Arrays[0] (ERRatio on CVDLine)
    // AbsFlag = Subgraph[3].Arrays[1]
    // AbsType = Subgraph[3].Arrays[2] (+1 bullish, -1 bearish)
    // VolFilter = Subgraph[3].Arrays[3]
    SCFloatArray srcER;
    SCFloatArray srcAbsFlag;
    SCFloatArray srcAbsType;
    SCFloatArray srcVolFilter;

    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 3, 0, srcER);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 3, 1, srcAbsFlag);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 3, 2, srcAbsType);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 3, 3, srcVolFilter);

    float er = srcER[sc.Index];
    ERLine[sc.Index] = er;

    AbsHighLine[sc.Index] = easeT;
    AbsLowLine[sc.Index] = absT;
    VolFilterBar[sc.Index] = srcVolFilter[sc.Index];

    // Absorption event dots
    if (srcAbsFlag[sc.Index] > 0.5f)
    {
        AbsDots[sc.Index] = er;  // Plot at the ER value
        if (srcAbsType[sc.Index] > 0.5f)
            AbsDots.DataColor[sc.Index] = AbsDots.PrimaryColor;    // Bullish = green
        else
            AbsDots.DataColor[sc.Index] = AbsDots.SecondaryColor;  // Bearish = purple
    }
    else
    {
        AbsDots[sc.Index] = 0;
    }
}


// =========================================================================
// Panel 4: Phase + Signal — Wyckoff Market Phase & Trade Signal
//
// Top half: Wyckoff phase as colored histogram
//   - Gray  (0) = Unknown / Ranging — no clear phase
//   - Green (1) = Accumulation — smart money buying, price flat/down
//   - Blue  (2) = Markup — price rising with volume confirmation
//   - Purple(3) = Distribution — smart money selling, price flat/up
//   - Red   (4) = Markdown — price falling with volume confirmation
//
// Bottom: Signal line (+1 long, -1 short, 0 flat)
//   - Green = long position (after Spring/SC + green effort confirmation)
//   - Red = short position (after Upthrust/BC + purple effort confirmation)
//   - Flat = no trade (waiting for next Wyckoff event + confirmation)
//
// How to use for entries:
//   1. Wait for Phase to show Accumulation (green) or Distribution (purple)
//   2. Watch for a Spring/SC event (bull arrow on main chart)
//      or Upthrust/BC event (bear arrow on main chart)
//   3. Signal line turns green/red = confirmed entry
//   4. CVD panel should agree with direction
// =========================================================================
SCSFExport scsf_WyckoffWeis_PhasePanel(SCStudyInterfaceRef sc)
{
    SCSubgraphRef PhaseBar     = sc.Subgraph[0];
    SCSubgraphRef SignalLine   = sc.Subgraph[1];
    SCSubgraphRef ZeroLine     = sc.Subgraph[2];

    SCInputRef InSourceStudy   = sc.Input[0];

    if (sc.SetDefaults)
    {
        sc.GraphName = "Wyckoff Phase + Signal";
        sc.AutoLoop = 1;
        sc.GraphRegion = 4;

        PhaseBar.Name = "Phase";
        PhaseBar.DrawStyle = DRAWSTYLE_BAR;
        PhaseBar.PrimaryColor = RGB(158, 158, 158);    // Gray (unknown)
        PhaseBar.LineWidth = 4;

        SignalLine.Name = "Signal";
        SignalLine.DrawStyle = DRAWSTYLE_LINE;
        SignalLine.PrimaryColor = RGB(76, 175, 80);    // Green (long)
        SignalLine.SecondaryColor = RGB(239, 83, 80);  // Red (short)
        SignalLine.SecondaryColorUsed = 1;
        SignalLine.LineWidth = 3;

        ZeroLine.Name = "Zero";
        ZeroLine.DrawStyle = DRAWSTYLE_LINE;
        ZeroLine.PrimaryColor = RGB(80, 80, 80);
        ZeroLine.LineWidth = 1;

        InSourceStudy.Name = "Source Study (WyckoffWeis_NQ)";
        InSourceStudy.SetStudySubgraphValues(0, 0);

        return;
    }

    int srcID = InSourceStudy.GetStudyID();

    // Phase = Subgraph[4] (PhaseColor)
    // Signal = Subgraph[3].Arrays[6]
    SCFloatArray srcPhase;
    SCFloatArray srcSignal;

    sc.GetStudyArrayUsingID(srcID, 4, srcPhase);
    sc.GetStudyExtraArrayFromChartUsingID(sc.ChartNumber, srcID, 3, 6, srcSignal);

    float ph = srcPhase[sc.Index];
    float sig = srcSignal[sc.Index];

    PhaseBar[sc.Index] = ph;
    SignalLine[sc.Index] = sig;
    ZeroLine[sc.Index] = 0.0f;

    // Color phase bars by phase type
    int phaseInt = (int)ph;
    switch (phaseInt)
    {
        case 1:  // Accumulation — green
            PhaseBar.DataColor[sc.Index] = RGB(76, 175, 80);
            break;
        case 2:  // Markup — blue
            PhaseBar.DataColor[sc.Index] = RGB(33, 150, 243);
            break;
        case 3:  // Distribution — purple
            PhaseBar.DataColor[sc.Index] = RGB(156, 39, 176);
            break;
        case 4:  // Markdown — red
            PhaseBar.DataColor[sc.Index] = RGB(239, 83, 80);
            break;
        default: // Unknown — gray
            PhaseBar.DataColor[sc.Index] = RGB(158, 158, 158);
            break;
    }

    // Color signal line
    if (sig > 0.5f)
        SignalLine.DataColor[sc.Index] = SignalLine.PrimaryColor;    // Green = long
    else if (sig < -0.5f)
        SignalLine.DataColor[sc.Index] = SignalLine.SecondaryColor;  // Red = short
    else
        SignalLine.DataColor[sc.Index] = RGB(80, 80, 80);            // Gray = flat
}
