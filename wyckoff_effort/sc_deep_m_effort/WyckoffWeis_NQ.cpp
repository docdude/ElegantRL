// ==========================================================================
// Wyckoff Weis Wave NQ — Sierra Chart Advanced Custom Study (ACSIL)
// File: WyckoffWeis_NQ.cpp
//
// Chart Setup: NQ futures, Range 40 (or any range/number bar type)
// Compile: Analysis > Build Custom Studies DLL > Remote Build
//
// Ported from: Wyckoff_wave_chatbot/utils/wyckoff_analyzer.py
//              + utils/models_utils/ww_utils.py (Weis Wave segmentation)
//
// This study implements:
//   1. Delta & Cumulative Volume Delta (CVD)
//   2. Weis Wave segmentation (ZigZag reversal-based)
//   3. Per-wave and per-bar Effort vs Result scoring
//   4. Absorption detection (high effort, low result)
//   5. Wyckoff events: Spring, Upthrust, Selling Climax, Buying Climax
//   6. Supply-demand phase classification (Accum/Markup/Dist/Markdown)
//   7. Volume & Time strength tiers (Normalized Emphasized, 5 levels)
//   8. Combined signal generation with zone rectangles
//
// Zone colors (matching DeepCharts convention):
//   Green  = bullish (accumulation/markup, path of least resistance UP)
//   Purple = bearish (distribution/markdown, path of least resistance DOWN)
//   Yellow = large wave / climax event markers
// ==========================================================================

#include "sierrachart.h"
#include <cmath>
#include <algorithm>

SCDLLName("WyckoffWeis_NQ")

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------
static float CalcEMA(float val, float prev, int period)
{
    float k = 2.0f / (period + 1.0f);
    return val * k + prev * (1.0f - k);
}

// Linear regression slope over last `len` values ending at arr[idx].
// Returns slope in value-per-bar units.
static float LinRegSlope(SCFloatArrayRef arr, int idx, int len)
{
    if (len < 2 || idx < len - 1) return 0.0f;
    // Using simplified least-squares: slope = 12*S1 / (n*(n^2-1))
    // where S1 = sum( (i - (n-1)/2) * y[i] )
    float sumXY = 0.0f, sumX2 = 0.0f;
    float mid = (len - 1) * 0.5f;
    for (int k = 0; k < len; k++)
    {
        float x = k - mid;
        sumXY += x * arr[idx - len + 1 + k];
        sumX2 += x * x;
    }
    return (sumX2 > 0.0f) ? sumXY / sumX2 : 0.0f;
}

// =========================================================================
// Main Study Function
// =========================================================================
SCSFExport scsf_WyckoffWeis_NQ(SCStudyInterfaceRef sc)
{
    // ----- Subgraphs (price chart overlay, Region 0) -----
    SCSubgraphRef WaveTrend    = sc.Subgraph[0];  // Weis wave trendline
    SCSubgraphRef BullArrow    = sc.Subgraph[1];  // Bullish event arrows
    SCSubgraphRef BearArrow    = sc.Subgraph[2];  // Bearish event arrows
    SCSubgraphRef CVDLine      = sc.Subgraph[3];  // CVD (hidden, for spreadsheet)
    SCSubgraphRef PhaseColor   = sc.Subgraph[4];  // Phase (hidden, for spreadsheet)
    SCSubgraphRef WaveVolSg    = sc.Subgraph[5];  // Wave volume (hidden)
    SCSubgraphRef EffortRatio  = sc.Subgraph[6];  // Effort/Result ratio (hidden)
    SCSubgraphRef VolStrength  = sc.Subgraph[7];  // Volume strength tier (hidden)

    // ----- Zone visibility/color subgraphs (not plotted, config only) -----
    SCSubgraphRef BullZoneRef   = sc.Subgraph[8];   // Green zone color/toggle
    SCSubgraphRef BearZoneRef   = sc.Subgraph[9];   // Purple zone color/toggle
    SCSubgraphRef YellowWaveRef = sc.Subgraph[10];  // Yellow wave color/toggle
    SCSubgraphRef AbsZoneRef    = sc.Subgraph[11];  // Absorption zone color/toggle

    // ----- Extra arrays on WaveTrend for intermediate data -----
    // [0] = Delta per bar
    // [1] = CVD (cumulative)
    // [2] = WaveDir (+1/-1)
    // [3] = WaveID
    // [4] = WaveHigh
    // [5] = WaveLow
    // [6] = WaveVolume (cumulative within wave)
    // [7] = WaveDelta (cumulative within wave)
    SCFloatArrayRef DeltaArr   = WaveTrend.Arrays[0];
    SCFloatArrayRef CVDArr     = WaveTrend.Arrays[1];
    SCFloatArrayRef WaveDir    = WaveTrend.Arrays[2];
    SCFloatArrayRef WaveID     = WaveTrend.Arrays[3];
    SCFloatArrayRef WaveHigh   = WaveTrend.Arrays[4];
    SCFloatArrayRef WaveLow    = WaveTrend.Arrays[5];
    SCFloatArrayRef WaveVol    = WaveTrend.Arrays[6];
    SCFloatArrayRef WaveDelt   = WaveTrend.Arrays[7];

    // Extra arrays on CVDLine for more intermediate data
    // [0] = ER_Ratio (effort/result normalized)
    // [1] = Absorption flag
    // [2] = AbsorptionType (+1 bullish, -1 bearish)
    // [3] = Volume filter (Normalized_Emphasized)
    // [4] = Time filter (Normalized_Emphasized)
    // [5] = Phase (0-4)
    // [6] = Signal (+1/-1/0)
    // [7] = BarDuration (seconds, from num_trades proxy)
    SCFloatArrayRef ERRatio    = CVDLine.Arrays[0];
    SCFloatArrayRef AbsFlag    = CVDLine.Arrays[1];
    SCFloatArrayRef AbsType    = CVDLine.Arrays[2];
    SCFloatArrayRef VolFilter  = CVDLine.Arrays[3];
    SCFloatArrayRef TimeFilter = CVDLine.Arrays[4];
    SCFloatArrayRef Phase      = CVDLine.Arrays[5];
    SCFloatArrayRef Signal     = CVDLine.Arrays[6];
    SCFloatArrayRef BarDur     = CVDLine.Arrays[7];

    // ----- Inputs -----
    SCInputRef InReversalPct       = sc.Input[0];
    SCInputRef InERLookback        = sc.Input[1];
    SCInputRef InAbsVolThresh      = sc.Input[2];
    SCInputRef InAbsERThresh       = sc.Input[3];
    SCInputRef InSwingLookback     = sc.Input[4];
    SCInputRef InEventVolMult      = sc.Input[5];
    SCInputRef InPhaseLookback     = sc.Input[6];
    SCInputRef InLargeWaveRatio    = sc.Input[7];
    SCInputRef InZoneExtension     = sc.Input[8];
    SCInputRef InVolNormPeriod     = sc.Input[9];
    SCInputRef InCooldownBars      = sc.Input[10];
    SCInputRef InConfirmBars       = sc.Input[11];
    SCInputRef InReversalMode      = sc.Input[12]; // 0=Percentage, 1=Points
    SCInputRef InUseChartSession    = sc.Input[13]; // Yes/No
    SCInputRef InCustomRTHStart     = sc.Input[14]; // Time
    SCInputRef InCustomRTHEnd       = sc.Input[15]; // Time
    SCInputRef InMinSignalQuality    = sc.Input[16]; // 0-100
    SCInputRef InShowBullZones       = sc.Input[17]; // Yes/No
    SCInputRef InShowBearZones       = sc.Input[18]; // Yes/No
    SCInputRef InShowYellowWaves     = sc.Input[19]; // Yes/No
    SCInputRef InShowAbsZones        = sc.Input[20]; // Yes/No
    SCInputRef InZoneTransparency    = sc.Input[21]; // 0-100

    // =====================================================================
    // DEFAULTS
    // =====================================================================
    if (sc.SetDefaults)
    {
        sc.GraphName = "Wyckoff Weis Wave NQ";
        sc.AutoLoop = 0;
        sc.GraphRegion = 0;
        sc.ScaleRangeType = SCALE_SAMEASREGION;

        // --- Weis wave trendline (ZigZag on price chart) ---
        WaveTrend.Name = "Wave Trendline";
        WaveTrend.DrawStyle = DRAWSTYLE_LINE_SKIP_ZEROS;
        WaveTrend.PrimaryColor = RGB(33, 150, 243);    // Blue
        WaveTrend.SecondaryColor = RGB(255, 152, 0);    // Orange
        WaveTrend.SecondaryColorUsed = 1;
        WaveTrend.LineWidth = 2;
        WaveTrend.DrawZeros = 0;

        // --- Signal arrows ---
        BullArrow.Name = "Bull Event";
        BullArrow.DrawStyle = DRAWSTYLE_ARROW_UP;
        BullArrow.PrimaryColor = RGB(0, 230, 118);
        BullArrow.LineWidth = 4;
        BullArrow.DrawZeros = 0;

        BearArrow.Name = "Bear Event";
        BearArrow.DrawStyle = DRAWSTYLE_ARROW_DOWN;
        BearArrow.PrimaryColor = RGB(255, 23, 68);
        BearArrow.LineWidth = 4;
        BearArrow.DrawZeros = 0;

        // --- Hidden subgraphs (spreadsheet / companion study access) ---
        CVDLine.Name = "CVD";
        CVDLine.DrawStyle = DRAWSTYLE_IGNORE;

        PhaseColor.Name = "Phase";
        PhaseColor.DrawStyle = DRAWSTYLE_IGNORE;

        WaveVolSg.Name = "Wave Volume";
        WaveVolSg.DrawStyle = DRAWSTYLE_IGNORE;

        EffortRatio.Name = "Effort/Result";
        EffortRatio.DrawStyle = DRAWSTYLE_IGNORE;

        VolStrength.Name = "Volume Strength";
        VolStrength.DrawStyle = DRAWSTYLE_IGNORE;

        // --- Zone color subgraphs (color only — visibility via Inputs) ---
        // Change zone colors via the color pickers on these subgraphs.
        // Toggle visibility via the Show ____ Zones inputs below.
        BullZoneRef.Name = "Bull Zone Color";
        BullZoneRef.DrawStyle = DRAWSTYLE_IGNORE;
        BullZoneRef.PrimaryColor = RGB(76, 175, 80);   // Green
        BullZoneRef.DrawZeros = 0;

        BearZoneRef.Name = "Bear Zone Color";
        BearZoneRef.DrawStyle = DRAWSTYLE_IGNORE;
        BearZoneRef.PrimaryColor = RGB(156, 39, 176);  // Purple
        BearZoneRef.DrawZeros = 0;

        YellowWaveRef.Name = "Large Wave Color";
        YellowWaveRef.DrawStyle = DRAWSTYLE_IGNORE;
        YellowWaveRef.PrimaryColor = RGB(255, 235, 59); // Yellow
        YellowWaveRef.DrawZeros = 0;

        AbsZoneRef.Name = "Absorption Zone Color";
        AbsZoneRef.DrawStyle = DRAWSTYLE_IGNORE;
        AbsZoneRef.PrimaryColor = RGB(76, 175, 80);   // Green (bull abs)
        AbsZoneRef.SecondaryColor = RGB(156, 39, 176); // Purple (bear abs)
        AbsZoneRef.SecondaryColorUsed = 1;
        AbsZoneRef.DrawZeros = 0;

        // --- Inputs ---
        InReversalPct.Name = "Wave Reversal % (or Points)";
        InReversalPct.SetFloat(40.0f);
        InReversalPct.SetFloatLimits(0.05f, 500.0f);

        InERLookback.Name = "Effort/Result Lookback";
        InERLookback.SetInt(20);
        InERLookback.SetIntLimits(5, 100);

        InAbsVolThresh.Name = "Absorption Vol Threshold (×avg)";
        InAbsVolThresh.SetFloat(1.5f);
        InAbsVolThresh.SetFloatLimits(1.0f, 5.0f);

        InAbsERThresh.Name = "Absorption ER Threshold (max)";
        InAbsERThresh.SetFloat(0.5f);
        InAbsERThresh.SetFloatLimits(0.1f, 2.0f);

        InSwingLookback.Name = "Swing Lookback (Spring/UT)";
        InSwingLookback.SetInt(20);
        InSwingLookback.SetIntLimits(5, 100);

        InEventVolMult.Name = "Event Volume Multiplier";
        InEventVolMult.SetFloat(2.0f);
        InEventVolMult.SetFloatLimits(1.0f, 5.0f);

        InPhaseLookback.Name = "Phase Classification Lookback";
        InPhaseLookback.SetInt(50);
        InPhaseLookback.SetIntLimits(10, 200);

        InLargeWaveRatio.Name = "Large Wave Ratio (×avg4)";
        InLargeWaveRatio.SetFloat(1.5f);
        InLargeWaveRatio.SetFloatLimits(1.0f, 5.0f);

        InZoneExtension.Name = "Zone Extension (bars)";
        InZoneExtension.SetInt(15);
        InZoneExtension.SetIntLimits(1, 100);

        InVolNormPeriod.Name = "Vol/Time Norm Period";
        InVolNormPeriod.SetInt(20);
        InVolNormPeriod.SetIntLimits(5, 100);

        InCooldownBars.Name = "Signal Cooldown (bars)";
        InCooldownBars.SetInt(5);
        InCooldownBars.SetIntLimits(1, 50);

        InConfirmBars.Name = "Signal Confirm Bars";
        InConfirmBars.SetInt(2);
        InConfirmBars.SetIntLimits(1, 5);

        InReversalMode.Name = "Reversal Mode (0=Pct, 1=Points)";
        InReversalMode.SetInt(1);
        InReversalMode.SetIntLimits(0, 1);

        InUseChartSession.Name = "Use Chart Session Times";
        InUseChartSession.SetYesNo(1);

        InCustomRTHStart.Name = "Custom RTH Start (if not chart)";
        InCustomRTHStart.SetTime(HMS_TIME(9, 30, 0));

        InCustomRTHEnd.Name = "Custom RTH End (if not chart)";
        InCustomRTHEnd.SetTime(HMS_TIME(16, 0, 0));

        InMinSignalQuality.Name = "Min Signal Quality (0-100, 0=all)";
        InMinSignalQuality.SetInt(0);
        InMinSignalQuality.SetIntLimits(0, 100);

        InShowBullZones.Name = "Show Bull Zones (Spring/SC)";
        InShowBullZones.SetYesNo(1);

        InShowBearZones.Name = "Show Bear Zones (UT/BC)";
        InShowBearZones.SetYesNo(1);

        InShowYellowWaves.Name = "Show Large Wave Markers";
        InShowYellowWaves.SetYesNo(1);

        InShowAbsZones.Name = "Show Absorption Zones";
        InShowAbsZones.SetYesNo(1);

        InZoneTransparency.Name = "Zone Transparency (0=opaque, 100=invisible)";
        InZoneTransparency.SetInt(80);
        InZoneTransparency.SetIntLimits(0, 100);

        return;
    }

    // =====================================================================
    // Read inputs
    // =====================================================================
    float reversalVal    = InReversalPct.GetFloat();
    int   reversalMode   = InReversalMode.GetInt();
    int   erLookback     = InERLookback.GetInt();
    float absVolThresh   = InAbsVolThresh.GetFloat();
    float absERThresh    = InAbsERThresh.GetFloat();
    int   swingLookback  = InSwingLookback.GetInt();
    float eventVolMult   = InEventVolMult.GetFloat();
    int   phaseLookback  = InPhaseLookback.GetInt();
    float largeWaveRatio = InLargeWaveRatio.GetFloat();
    int   zoneExt        = InZoneExtension.GetInt();
    int   volNormPeriod  = InVolNormPeriod.GetInt();
    int   cooldownBars   = InCooldownBars.GetInt();
    int   confirmBars    = InConfirmBars.GetInt();

    float reversalThreshold = (reversalMode == 0)
        ? reversalVal / 100.0f   // percentage mode
        : reversalVal;           // points mode

    // Session time boundaries
    int sessionStart = InUseChartSession.GetYesNo() ? sc.StartTime1 : InCustomRTHStart.GetTime();
    int sessionEnd   = InUseChartSession.GetYesNo() ? sc.EndTime1   : InCustomRTHEnd.GetTime();
    int minQuality   = InMinSignalQuality.GetInt();
    int showBullZones   = InShowBullZones.GetYesNo();
    int showBearZones   = InShowBearZones.GetYesNo();
    int showYellowWaves = InShowYellowWaves.GetYesNo();
    int showAbsZones    = InShowAbsZones.GetYesNo();
    int zoneTrans       = InZoneTransparency.GetInt();

    // =====================================================================
    // PASS 1: Per-bar metrics (delta, CVD, E/R, absorption, events, strength)
    // =====================================================================

    // --- Weis Wave state (persistent across updates via subgraph arrays) ---
    // On full recalculation, initialize from bar 0.
    // On live updates, re-read state from the previous bar's arrays.

    int waveDir_i = 1;         // current wave direction: +1 up, -1 down
    int waveID_i  = 0;
    float wHigh   = 0.0f;
    float wLow    = 99999999.0f;
    float wVol    = 0.0f;
    float wDelt   = 0.0f;
    int extremumIdx = 0;
    float extremumPrice = 0.0f;

    int startIdx = sc.UpdateStartIndex;

    if (startIdx > 0)
    {
        // Restore wave state from previous bar
        int prev = startIdx - 1;
        waveDir_i = (int)WaveDir[prev];
        waveID_i  = (int)WaveID[prev];
        wHigh     = WaveHigh[prev];
        wLow      = WaveLow[prev];
        wVol      = WaveVol[prev];
        wDelt     = WaveDelt[prev];
        // Find extremum index by scanning back within current wave
        extremumIdx = prev;
        if (waveDir_i == 1)
        {
            extremumPrice = wHigh;
            for (int k = prev; k >= 0 && (int)WaveID[k] == waveID_i; k--)
            {
                if (sc.BaseData[SC_HIGH][k] >= extremumPrice)
                {
                    extremumPrice = sc.BaseData[SC_HIGH][k];
                    extremumIdx = k;
                }
            }
        }
        else
        {
            extremumPrice = wLow;
            for (int k = prev; k >= 0 && (int)WaveID[k] == waveID_i; k--)
            {
                if (sc.BaseData[SC_LOW][k] <= extremumPrice)
                {
                    extremumPrice = sc.BaseData[SC_LOW][k];
                    extremumIdx = k;
                }
            }
        }
    }

    for (int idx = startIdx; idx < sc.ArraySize; ++idx)
    {
        float open     = sc.BaseData[SC_OPEN][idx];
        float high     = sc.BaseData[SC_HIGH][idx];
        float low      = sc.BaseData[SC_LOW][idx];
        float close    = sc.BaseData[SC_LAST][idx];
        float volume   = sc.BaseData[SC_VOLUME][idx];
        float askVol   = sc.BaseData[SC_ASKVOL][idx];
        float bidVol   = sc.BaseData[SC_BIDVOL][idx];
        float nTrades  = sc.BaseData[SC_NUM_TRADES][idx];

        // Zero signal arrays
        BullArrow[idx] = 0;
        BearArrow[idx] = 0;

        // -----------------------------------------------------------
        // 1. Delta & CVD
        // -----------------------------------------------------------
        float delta = askVol - bidVol;
        DeltaArr[idx] = delta;
        CVDArr[idx] = (idx > 0) ? CVDArr[idx - 1] + delta : delta;
        CVDLine[idx] = CVDArr[idx];

        // -----------------------------------------------------------
        // 2. Weis Wave segmentation (ZigZag state machine)
        // -----------------------------------------------------------

        // Detect session/day boundary using GetTradingDayStartDateTimeOfBar
        // and RTH time filtering.
        SCDateTime curTDStart  = sc.GetTradingDayStartDateTimeOfBar(sc.BaseDateTimeIn[idx]);
        SCDateTime prevTDStart = (idx > 0)
            ? sc.GetTradingDayStartDateTimeOfBar(sc.BaseDateTimeIn[idx - 1])
            : curTDStart;
        bool newTradingDay = (idx == 0 || curTDStart != prevTDStart);

        int barTime = sc.BaseDateTimeIn[idx].GetTime();
        bool inRTH = (barTime >= sessionStart && barTime <= sessionEnd);

        bool sessionBreak = newTradingDay;

        if (idx == 0 || sessionBreak)
        {
            waveDir_i = 1;
            waveID_i  = (idx == 0) ? 0 : waveID_i + 1;
            wHigh = close;
            wLow = close;
            wVol = volume;
            wDelt = delta;
            extremumIdx = idx;
            extremumPrice = close;
        }
        else
        {
            bool reversed = false;

            if (waveDir_i == 1)  // up wave
            {
                if (close > wHigh)
                {
                    wHigh = close;
                    extremumIdx = idx;
                    extremumPrice = wHigh;
                }
                // Check reversal
                if (reversalMode == 0)  // percentage
                    reversed = (extremumPrice > 0) &&
                        ((extremumPrice - close) / extremumPrice >= reversalThreshold);
                else  // points
                    reversed = (extremumPrice - close) >= reversalThreshold;

                if (reversed)
                {
                    // Start new down wave
                    waveDir_i = -1;
                    waveID_i++;
                    wHigh = close;
                    wLow = close;
                    wVol = volume;
                    wDelt = delta;
                    extremumIdx = idx;
                    extremumPrice = wLow;
                }
                else
                {
                    wVol += volume;
                    wDelt += delta;
                }
            }
            else  // down wave
            {
                if (close < wLow)
                {
                    wLow = close;
                    extremumIdx = idx;
                    extremumPrice = wLow;
                }
                // Check reversal
                if (reversalMode == 0)
                    reversed = (extremumPrice > 0) &&
                        ((close - extremumPrice) / extremumPrice >= reversalThreshold);
                else
                    reversed = (close - extremumPrice) >= reversalThreshold;

                if (reversed)
                {
                    // Start new up wave
                    waveDir_i = 1;
                    waveID_i++;
                    wHigh = close;
                    wLow = close;
                    wVol = volume;
                    wDelt = delta;
                    extremumIdx = idx;
                    extremumPrice = wHigh;
                }
                else
                {
                    wVol += volume;
                    wDelt += delta;
                }
            }
        }

        WaveDir[idx]  = (float)waveDir_i;
        WaveID[idx]   = (float)waveID_i;
        WaveHigh[idx] = wHigh;
        WaveLow[idx]  = wLow;
        WaveVol[idx]  = wVol;
        WaveDelt[idx] = wDelt;
        WaveVolSg[idx] = wVol;

        // Wave trendline data: store extremum price at pivot bars, 0 elsewhere.
        // Bars outside RTH or at session boundaries forced to 0 so
        // DRAWSTYLE_LINE_SKIP_ZEROS breaks the line across the gap.
        // The !sessionBreak guard ensures the first bar of each new
        // trading day is always 0, even if adjacent to the prior day's
        // last bar (RTH-only charts have no overnight bars to gap on).
        WaveTrend[idx] = 0;
        if (inRTH && idx == extremumIdx && !sessionBreak)
        {
            WaveTrend[idx] = extremumPrice;
        }

        // -----------------------------------------------------------
        // 3. Effort vs Result (per bar)
        //    EffortResult = |price_change| / volume
        //    ER_Ratio = current / rolling_mean (< 0.5 = absorption)
        // -----------------------------------------------------------
        float priceChange = (idx > 0)
            ? fabsf(close - sc.BaseData[SC_LAST][idx - 1]) : 0.0f;
        float effortResult = (volume > 0) ? priceChange / volume : 0.0f;
        EffortRatio[idx] = effortResult;

        // Rolling mean of EffortResult for normalization
        float erSum = 0.0f;
        int erCount = 0;
        int erStart = (idx - erLookback + 1 > 0) ? idx - erLookback + 1 : 0;
        for (int k = erStart; k <= idx; k++)
        {
            erSum += EffortRatio[k];
            erCount++;
        }
        float erMean = (erCount > 0) ? erSum / erCount : 1.0f;
        float erRatio = (erMean > 1e-12f) ? effortResult / erMean : 1.0f;
        ERRatio[idx] = erRatio;

        // -----------------------------------------------------------
        // 4. Volume & Time strength (Normalized Emphasized)
        //    volFilter = ((vol / rolling_avg) * 100 - 100) * multiplier
        //    5 tiers: 0=lowest, 1=low, 2=average, 3=high, 4=ultra
        // -----------------------------------------------------------
        // Bar duration proxy: for range/number bars, use volume/num_trades
        float barDur = (nTrades > 1) ? volume / nTrades : 1.0f;
        BarDur[idx] = barDur;

        // Rolling average volume
        float volAvg = 0.0f;
        float durAvg = 0.0f;
        int normStart = (idx - volNormPeriod + 1 > 0) ? idx - volNormPeriod + 1 : 0;
        int normCount = idx - normStart + 1;
        for (int k = normStart; k <= idx; k++)
        {
            volAvg += sc.BaseData[SC_VOLUME][k];
            durAvg += BarDur[k];
        }
        volAvg = (normCount > 0) ? volAvg / normCount : 1.0f;
        durAvg = (normCount > 0) ? durAvg / normCount : 1.0f;

        float volNorm = (volAvg > 0) ? volume / volAvg : 1.0f;
        float volPct = (volNorm * 100.0f) - 100.0f;
        VolFilter[idx] = fabsf(volPct);

        float durNorm = (durAvg > 0) ? barDur / durAvg : 1.0f;
        float durPct = (durNorm * 100.0f) - 100.0f;
        TimeFilter[idx] = fabsf(durPct);

        // Strength tier classification (Normalized Emphasized percentages)
        // Thresholds: lowest<23.6, low<38.2, average<61.8, high<100, ultra>=101
        int vsTier = 0;
        float vf = VolFilter[idx];
        if      (vf >= 101.0f) vsTier = 4;
        else if (vf >= 61.8f)  vsTier = 3;
        else if (vf >= 38.2f)  vsTier = 2;
        else if (vf >= 23.6f)  vsTier = 1;
        else                   vsTier = 0;
        VolStrength[idx] = (float)vsTier;

        // -----------------------------------------------------------
        // 5. Absorption detection
        //    High volume (>= absVolThresh × avg) + low ER_Ratio (<= absERThresh)
        // -----------------------------------------------------------
        float volRatio = (volAvg > 0) ? volume / volAvg : 1.0f;
        bool isAbsorption = (volRatio >= absVolThresh) && (erRatio <= absERThresh);
        AbsFlag[idx] = isAbsorption ? 1.0f : 0.0f;

        if (isAbsorption)
            AbsType[idx] = (delta > 0) ? 1.0f : -1.0f;  // +1 bullish, -1 bearish
        else
            AbsType[idx] = 0.0f;

        // -----------------------------------------------------------
        // 6. Wyckoff event detection (Spring, Upthrust, SC, BC)
        // -----------------------------------------------------------
        // Computed in Pass 2 (needs rolling swing high/low)
        // We pre-compute volume average for event detection here.

    }  // end Pass 1 loop

    // =====================================================================
    // ZIGZAG INTERPOLATION PASS: Fill every bar between consecutive pivot
    // points with linearly interpolated values so the subgraph line renders
    // as a continuous zigzag.  At session boundaries or outside RTH, leave
    // value = 0 so DRAWSTYLE_LINE_SKIP_ZEROS creates a visual break.
    // =====================================================================
    {
        int prevPivotIdx = -1;
        float prevPivotPrice = 0.0f;
        SCDateTime prevPivotTDStart;

        for (int idx = 0; idx < sc.ArraySize; ++idx)
        {
            if (WaveTrend[idx] != 0.0f)
            {
                SCDateTime thisTDStart = sc.GetTradingDayStartDateTimeOfBar(sc.BaseDateTimeIn[idx]);

                // This bar is a pivot point
                if (prevPivotIdx >= 0 && prevPivotTDStart == thisTDStart)
                {
                    // Linearly interpolate all bars between prev pivot
                    // and this pivot (exclusive of endpoints).
                    int span = idx - prevPivotIdx;
                    if (span > 1)
                    {
                        for (int k = prevPivotIdx + 1; k < idx; k++)
                        {
                            // Check this bar is still in the same session & RTH
                            SCDateTime kTDStart = sc.GetTradingDayStartDateTimeOfBar(sc.BaseDateTimeIn[k]);
                            int kTime = sc.BaseDateTimeIn[k].GetTime();
                            bool kInRTH = (kTime >= sessionStart && kTime <= sessionEnd);
                            if (kTDStart != prevPivotTDStart || !kInRTH)
                                break;
                            float t = (float)(k - prevPivotIdx) / (float)span;
                            WaveTrend[k] = prevPivotPrice
                                + t * (WaveTrend[idx] - prevPivotPrice);
                            WaveTrend.DataColor[k] = (WaveDir[idx] > 0)
                                ? WaveTrend.PrimaryColor
                                : WaveTrend.SecondaryColor;
                        }
                    }
                }

                // Color the pivot bar itself
                WaveTrend.DataColor[idx] = (WaveDir[idx] > 0)
                    ? WaveTrend.PrimaryColor
                    : WaveTrend.SecondaryColor;

                prevPivotIdx = idx;
                prevPivotPrice = WaveTrend[idx];
                prevPivotTDStart = thisTDStart;
            }
        }

        // Fill from last confirmed pivot to end of array (live segment)
        if (prevPivotIdx >= 0 && prevPivotIdx < sc.ArraySize - 1)
        {
            int lastBar = sc.ArraySize - 1;
            SCDateTime lastTDStart = sc.GetTradingDayStartDateTimeOfBar(sc.BaseDateTimeIn[lastBar]);
            int lastTime = sc.BaseDateTimeIn[lastBar].GetTime();
            bool lastInRTH = (lastTime >= sessionStart && lastTime <= sessionEnd);

            if (lastTDStart == prevPivotTDStart && lastInRTH)
            {
                float runningPrice;
                if (WaveDir[lastBar] > 0)
                    runningPrice = WaveHigh[lastBar];
                else
                    runningPrice = WaveLow[lastBar];

                int span = lastBar - prevPivotIdx;
                if (span > 0)
                {
                    for (int k = prevPivotIdx + 1; k <= lastBar; k++)
                    {
                        SCDateTime kTDStart = sc.GetTradingDayStartDateTimeOfBar(sc.BaseDateTimeIn[k]);
                        int kTime = sc.BaseDateTimeIn[k].GetTime();
                        bool kInRTH = (kTime >= sessionStart && kTime <= sessionEnd);
                        if (kTDStart != prevPivotTDStart || !kInRTH)
                            break;
                        float t = (float)(k - prevPivotIdx) / (float)span;
                        WaveTrend[k] = prevPivotPrice
                            + t * (runningPrice - prevPivotPrice);
                        WaveTrend.DataColor[k] = (WaveDir[lastBar] > 0)
                            ? WaveTrend.PrimaryColor
                            : WaveTrend.SecondaryColor;
                    }
                }
            }
        }
    }

    // =====================================================================
    // PASS 2: Wyckoff events, Phase classification, Signals, Zones
    // =====================================================================
    {
        // We need swing high/low and rolling averages that require all
        // Pass 1 data. For live updates, only reclassify recent bars.
        int pass2Start;
        int reScanLookback = phaseLookback + swingLookback + 20;
        if (sc.UpdateStartIndex == 0)
        {
            pass2Start = swingLookback + 1;

            // Clear all zone drawings on full recalc
            for (int ln = 70000; ln < 70000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
            for (int ln = 80000; ln < 80000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
        }
        else
        {
            pass2Start = sc.UpdateStartIndex - reScanLookback;
            if (pass2Start < swingLookback + 1)
                pass2Start = swingLookback + 1;

            for (int ln = 70000 + pass2Start; ln < 70000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
            for (int ln = 80000 + pass2Start; ln < 80000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
        }

        // ---- Wave comparison state (4-wave sliding window) ----
        float prevWaveVol[4]  = {0, 0, 0, 0};
        float prevWaveER[4]   = {0, 0, 0, 0};
        float prevUpVol = 0.0f, prevUpER = 0.0f;
        float prevDownVol = 0.0f, prevDownER = 0.0f;
        int lastWaveID = -1;

        // Pre-populate wave comparison state from waves before pass2Start
        if (pass2Start > 0)
        {
            // Walk through all bars before pass2Start to build wave stats
            int prevID = -1;
            int waveCount = 0;
            for (int k = 0; k < pass2Start; k++)
            {
                int curID = (int)WaveID[k];
                if (curID != prevID && prevID >= 0)
                {
                    // Wave prevID just completed at k-1
                    float wv = WaveVol[k - 1];
                    float wd = (float)(int)WaveDir[k - 1];
                    float wh = WaveHigh[k - 1];
                    float wl = WaveLow[k - 1];
                    float disp = fabsf(wh - wl);
                    float wer = (disp > 0) ? wv / disp : 0.0f;

                    // Shift sliding window
                    prevWaveVol[0] = prevWaveVol[1];
                    prevWaveVol[1] = prevWaveVol[2];
                    prevWaveVol[2] = prevWaveVol[3];
                    prevWaveVol[3] = wv;
                    prevWaveER[0] = prevWaveER[1];
                    prevWaveER[1] = prevWaveER[2];
                    prevWaveER[2] = prevWaveER[3];
                    prevWaveER[3] = wer;

                    if ((int)wd == 1) { prevUpVol = wv; prevUpER = wer; }
                    else              { prevDownVol = wv; prevDownER = wer; }

                    waveCount++;
                }
                prevID = curID;
                lastWaveID = curID;
            }
        }

        // ---- Signal state ----
        int currentSignal = 0;
        int barsSinceChange = cooldownBars + 1; // start ready

        // Restore signal state if continuing
        if (pass2Start > 0)
        {
            for (int k = pass2Start - 1; k >= 0; k--)
            {
                if (Signal[k] != 0)
                {
                    currentSignal = (int)Signal[k];
                    barsSinceChange = pass2Start - k;
                    break;
                }
            }
        }

        for (int idx = pass2Start; idx < sc.ArraySize; ++idx)
        {
            float close  = sc.BaseData[SC_LAST][idx];
            float high   = sc.BaseData[SC_HIGH][idx];
            float low    = sc.BaseData[SC_LOW][idx];
            float open   = sc.BaseData[SC_OPEN][idx];
            float volume = sc.BaseData[SC_VOLUME][idx];
            float delta  = DeltaArr[idx];
            float erRatio = ERRatio[idx];
            float volRatio = 0.0f;
            {
                // Recompute vol ratio for event detection
                float va = 0.0f;
                int ns = (idx - swingLookback + 1 > 0) ? idx - swingLookback + 1 : 0;
                int nc = idx - ns + 1;
                for (int k = ns; k <= idx; k++)
                    va += sc.BaseData[SC_VOLUME][k];
                va = (nc > 0) ? va / nc : 1.0f;
                volRatio = (va > 0) ? volume / va : 1.0f;
            }

            barsSinceChange++;

            // --- Wave completion detection ---
            int curID = (int)WaveID[idx];
            bool waveJustCompleted = false;
            int completedWaveLastBar = -1;
            if (curID != lastWaveID && lastWaveID >= 0 && idx > 0)
            {
                waveJustCompleted = true;
                completedWaveLastBar = idx - 1;
            }

            if (waveJustCompleted && completedWaveLastBar >= 0)
            {
                float wv  = WaveVol[completedWaveLastBar];
                float wd  = WaveDir[completedWaveLastBar];
                float wh  = WaveHigh[completedWaveLastBar];
                float wl  = WaveLow[completedWaveLastBar];
                float disp = fabsf(wh - wl);
                float wer  = (disp > 0) ? wv / disp : 0.0f;

                // Large wave detection
                bool allNonZero = (prevWaveVol[0] > 0 && prevWaveVol[1] > 0 &&
                                   prevWaveVol[2] > 0 && prevWaveVol[3] > 0);
                if (allNonZero)
                {
                    float avg4vol = (prevWaveVol[0] + prevWaveVol[1] +
                                     prevWaveVol[2] + prevWaveVol[3]) / 4.0f;
                    if (wv > avg4vol * largeWaveRatio)
                    {
                        // Draw only if Yellow Wave subgraph is not hidden
                        if (showYellowWaves)
                        {
                        // Mark large wave — thin yellow band at the wave's
                        // turning point (last 3 bars of the wave), not the
                        // entire wave span which would create huge boxes.
                        s_UseTool tool;
                        tool.Clear();
                        tool.ChartNumber = sc.ChartNumber;
                        tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
                        tool.LineNumber = 80000 + completedWaveLastBar;
                        tool.AddMethod = UTAM_ADD_OR_ADJUST;

                        // Narrow band: last 3 bars of the wave
                        int bandStart = completedWaveLastBar - 2;
                        if (bandStart < 0) bandStart = 0;
                        tool.BeginIndex = bandStart;
                        tool.EndIndex   = completedWaveLastBar + 3;

                        // Use the turning point bar's body ± 1 tick
                        // instead of the full wave high/low
                        float tpOpen  = sc.BaseData[SC_OPEN][completedWaveLastBar];
                        float tpClose = sc.BaseData[SC_LAST][completedWaveLastBar];
                        float bodyHi  = (tpOpen > tpClose) ? tpOpen : tpClose;
                        float bodyLo  = (tpOpen < tpClose) ? tpOpen : tpClose;
                        tool.BeginValue = bodyLo - sc.TickSize * 2;
                        tool.EndValue   = bodyHi + sc.TickSize * 2;

                        tool.Color = YellowWaveRef.PrimaryColor;
                        tool.SecondaryColor = tool.Color;
                        tool.TransparencyLevel = zoneTrans;
                        tool.AddAsUserDrawnDrawing = 0;
                        sc.UseTool(tool);
                        }
                    }
                }

                // Update sliding window
                prevWaveVol[0] = prevWaveVol[1];
                prevWaveVol[1] = prevWaveVol[2];
                prevWaveVol[2] = prevWaveVol[3];
                prevWaveVol[3] = wv;
                prevWaveER[0]  = prevWaveER[1];
                prevWaveER[1]  = prevWaveER[2];
                prevWaveER[2]  = prevWaveER[3];
                prevWaveER[3]  = wer;

                if ((int)wd == 1) { prevUpVol = wv; prevUpER = wer; }
                else              { prevDownVol = wv; prevDownER = wer; }
            }
            lastWaveID = curID;

            // -----------------------------------------------------------
            // 6. Wyckoff events: Spring, Upthrust, SC, BC
            // -----------------------------------------------------------
            bool isSpring = false, isUpthrust = false;
            bool isSC = false, isBC = false;

            if (idx >= swingLookback + 1)
            {
                // Rolling swing high/low
                float swingHigh = sc.BaseData[SC_HIGH][idx - 1];
                float swingLow  = sc.BaseData[SC_LOW][idx - 1];
                for (int k = idx - swingLookback; k < idx; k++)
                {
                    if (sc.BaseData[SC_HIGH][k] > swingHigh)
                        swingHigh = sc.BaseData[SC_HIGH][k];
                    if (sc.BaseData[SC_LOW][k] < swingLow)
                        swingLow = sc.BaseData[SC_LOW][k];
                }

                float barRange = high - low;
                bool volSpike = (volRatio >= eventVolMult);

                if (barRange > 0)
                {
                    // Selling Climax: large down bar, extreme vol, close near low,
                    // but delta positive (buyers absorbing)
                    bool priceDown = (close < sc.BaseData[SC_LAST][idx - 1]);
                    bool closeNearLow = ((close - low) / barRange < 0.3f);
                    if (volSpike && priceDown && closeNearLow && delta > 0)
                        isSC = true;

                    // Buying Climax: large up bar, extreme vol, close near high,
                    // but delta negative (sellers absorbing)
                    bool priceUp = (close > sc.BaseData[SC_LAST][idx - 1]);
                    bool closeNearHigh = ((high - close) / barRange < 0.3f);
                    if (volSpike && priceUp && closeNearHigh && delta < 0)
                        isBC = true;
                }

                // Spring: low breaks below swing low, close recovers above,
                // delta positive (buyers defending)
                if (low < swingLow && close > swingLow && delta > 0)
                    isSpring = true;

                // Upthrust: high breaks above swing high, close falls below,
                // delta negative (sellers defending)
                if (high > swingHigh && close < swingHigh && delta < 0)
                    isUpthrust = true;
            }

            // -----------------------------------------------------------
            // 7. Phase classification
            //    0=Unknown, 1=Accumulation, 2=Markup, 3=Distribution, 4=Markdown
            // -----------------------------------------------------------
            int phase = 0;
            if (idx >= phaseLookback)
            {
                // Price slope (linear regression over phaseLookback bars)
                float priceSlope = 0.0f;
                {
                    float sumXY = 0.0f, sumX2 = 0.0f;
                    float mid = (phaseLookback - 1) * 0.5f;
                    for (int k = 0; k < phaseLookback; k++)
                    {
                        float x = k - mid;
                        sumXY += x * sc.BaseData[SC_LAST][idx - phaseLookback + 1 + k];
                        sumX2 += x * x;
                    }
                    priceSlope = (sumX2 > 0) ? sumXY / sumX2 : 0.0f;
                }

                // CVD slope
                float cvdSlope = LinRegSlope(CVDArr, idx, phaseLookback);

                // Normalize price slope relative to price level
                float priceLevel = 0.0f;
                for (int k = idx - phaseLookback + 1; k <= idx; k++)
                    priceLevel += sc.BaseData[SC_LAST][k];
                priceLevel = (phaseLookback > 0) ? priceLevel / phaseLookback : 1.0f;
                float normPriceSlope = (priceLevel != 0) ? priceSlope / priceLevel : 0.0f;

                // Rolling absorption counts
                int bullAbs = 0, bearAbs = 0;
                for (int k = idx - phaseLookback + 1; k <= idx; k++)
                {
                    if (AbsType[k] > 0.5f)  bullAbs++;
                    if (AbsType[k] < -0.5f) bearAbs++;
                }

                // Classify
                if (normPriceSlope < 0.0002f && cvdSlope > 0 && bullAbs > bearAbs)
                    phase = 1;  // Accumulation
                else if (normPriceSlope > 0.0005f && cvdSlope > 0)
                    phase = 2;  // Markup
                else if (normPriceSlope > -0.0002f && cvdSlope < 0 && bearAbs > bullAbs)
                    phase = 3;  // Distribution
                else if (normPriceSlope < -0.0005f && cvdSlope < 0)
                    phase = 4;  // Markdown
            }

            Phase[idx] = (float)phase;
            PhaseColor[idx] = (float)phase;

            // -----------------------------------------------------------
            // 8. Effort-based signal score (basic effort from delta+CVD)
            //    Positive = green (bullish), Negative = purple (bearish)
            // -----------------------------------------------------------
            float effortScore = 0.0f;
            {
                // Delta ratio: delta / rolling avg volume
                float avgVol = 0.0f;
                int nss = (idx - erLookback + 1 > 0) ? idx - erLookback + 1 : 0;
                int ncc = idx - nss + 1;
                for (int k = nss; k <= idx; k++)
                    avgVol += sc.BaseData[SC_VOLUME][k];
                avgVol = (ncc > 0) ? avgVol / ncc : 1.0f;

                float deltaRatio = (avgVol > 0) ? delta / avgVol : 0.0f;
                if (deltaRatio > 1.0f) deltaRatio = 1.0f;
                if (deltaRatio < -1.0f) deltaRatio = -1.0f;

                // CVD fast/slow EMA slope
                float cvdFast = CVDArr[idx];
                float cvdSlow = CVDArr[idx];
                if (idx >= erLookback)
                {
                    // Simple approximation: avg of recent CVD vs avg of older CVD
                    float fastSum = 0.0f, slowSum = 0.0f;
                    int fastN = (erLookback < idx + 1) ? erLookback : idx + 1;
                    int slowN = (erLookback * 3 < idx + 1) ? erLookback * 3 : idx + 1;
                    for (int k = idx - fastN + 1; k <= idx; k++)
                        fastSum += CVDArr[k];
                    for (int k = idx - slowN + 1; k <= idx; k++)
                        slowSum += CVDArr[k];
                    cvdFast = fastSum / fastN;
                    cvdSlow = slowSum / slowN;
                }
                float cvdSlope = cvdFast - cvdSlow;

                // Scale and combine
                // Use tanh for bounding
                float scale = fabsf(cvdSlope) + 1e-6f;
                float cvdNorm = tanhf(cvdSlope / scale);

                effortScore = 0.6f * cvdNorm + 0.4f * deltaRatio;
            }

            // -----------------------------------------------------------
            // 9. Signal generation with Quality Score
            //    Long: Spring/SC + green effort + Accumulation context
            //    Short: Upthrust/BC + purple effort + Distribution context
            //
            //    Quality score (0-100) measures confluence:
            //      +20  Wyckoff event (Spring/UT/SC/BC)
            //      +20  Volume spike (volRatio >= eventVolMult)
            //      +15  Absorption on this bar or within last 3 bars
            //      +15  Phase context aligns (Accum for long, Dist for short)
            //      +15  Effort score confirms direction
            //      +15  CVD divergence from price (momentum confirmation)
            //    A prop trader would typically set threshold 50-70.
            // -----------------------------------------------------------
            // Count recent green/purple effort bars
            int greenCount = 0, purpleCount = 0;
            for (int k = (idx - 2 > 0 ? idx - 2 : 0); k <= idx; k++)
            {
                // Re-derive effort score inline for recent bars
                // (stored in Signal array would be circular, so use CVD-based proxy)
                float d = DeltaArr[k];
                float v = sc.BaseData[SC_VOLUME][k];
                float dr = 0.0f;
                {
                    float av = 0.0f;
                    int s2 = (k - erLookback + 1 > 0) ? k - erLookback + 1 : 0;
                    int c2 = k - s2 + 1;
                    for (int j = s2; j <= k; j++)
                        av += sc.BaseData[SC_VOLUME][j];
                    av = (c2 > 0) ? av / c2 : 1.0f;
                    dr = (av > 0) ? d / av : 0.0f;
                }
                if (dr > 0)  greenCount++;
                if (dr < 0)  purpleCount++;
            }

            // --- Compute signal quality score ---
            int bullQuality = 0, bearQuality = 0;

            // +20: Wyckoff event detected
            if (isSpring || isSC) bullQuality += 20;
            if (isUpthrust || isBC) bearQuality += 20;

            // +20: Volume spike
            if (volRatio >= eventVolMult)
            {
                if (isSpring || isSC) bullQuality += 20;
                if (isUpthrust || isBC) bearQuality += 20;
            }

            // +15: Absorption present (this bar or within last 3 bars)
            {
                bool recentBullAbs = false, recentBearAbs = false;
                int absStart = (idx - 3 > 0) ? idx - 3 : 0;
                for (int k = absStart; k <= idx; k++)
                {
                    if (AbsFlag[k] > 0.5f)
                    {
                        if (AbsType[k] > 0.5f)  recentBullAbs = true;
                        if (AbsType[k] < -0.5f) recentBearAbs = true;
                    }
                }
                if (recentBullAbs) bullQuality += 15;
                if (recentBearAbs) bearQuality += 15;
            }

            // +15: Phase context aligns
            if (phase == 1) bullQuality += 15;  // Accumulation → long
            if (phase == 3) bearQuality += 15;  // Distribution → short

            // +15: Effort score confirms direction
            if (effortScore > 0.15f)  bullQuality += 15;
            if (effortScore < -0.15f) bearQuality += 15;

            // +15: CVD slope confirms direction
            {
                float cvdSlope = 0.0f;
                if (idx >= 5)
                {
                    cvdSlope = CVDArr[idx] - CVDArr[idx - 5];
                }
                if (cvdSlope > 0) bullQuality += 15;
                if (cvdSlope < 0) bearQuality += 15;
            }

            // Determine effective quality for this bar's event direction
            int eventQuality = 0;
            bool isBullEvent = (isSpring || isSC);
            bool isBearEvent = (isUpthrust || isBC);
            if (isBullEvent) eventQuality = bullQuality;
            if (isBearEvent) eventQuality = bearQuality;
            // If both triggered (rare), take the higher
            if (isBullEvent && isBearEvent)
                eventQuality = (bullQuality > bearQuality) ? bullQuality : bearQuality;

            // Gate events by quality threshold
            bool qualifiedBull = isBullEvent && (bullQuality >= minQuality);
            bool qualifiedBear = isBearEvent && (bearQuality >= minQuality);

            // Entry logic
            if (currentSignal == 0 && barsSinceChange >= cooldownBars)
            {
                // Long entry
                if (qualifiedBull && greenCount >= confirmBars &&
                    (phase == 0 || phase == 1))
                {
                    currentSignal = 1;
                    barsSinceChange = 0;
                }
                // Short entry
                else if (qualifiedBear && purpleCount >= confirmBars &&
                         (phase == 0 || phase == 3))
                {
                    currentSignal = -1;
                    barsSinceChange = 0;
                }
            }
            // Exit / reversal logic (exits don't require quality gate)
            else if (currentSignal == 1 && barsSinceChange >= cooldownBars)
            {
                if (isUpthrust)
                    { currentSignal = -1; barsSinceChange = 0; }
                else if (isBC)
                    { currentSignal = 0; barsSinceChange = 0; }
                else if (phase == 3 && purpleCount >= confirmBars)
                    { currentSignal = 0; barsSinceChange = 0; }
                else if (purpleCount >= 3)
                    { currentSignal = 0; barsSinceChange = 0; }
            }
            else if (currentSignal == -1 && barsSinceChange >= cooldownBars)
            {
                if (isSpring)
                    { currentSignal = 1; barsSinceChange = 0; }
                else if (isSC)
                    { currentSignal = 0; barsSinceChange = 0; }
                else if (phase == 1 && greenCount >= confirmBars)
                    { currentSignal = 0; barsSinceChange = 0; }
                else if (greenCount >= 3)
                    { currentSignal = 0; barsSinceChange = 0; }
            }

            Signal[idx] = (float)currentSignal;

            // -----------------------------------------------------------
            // Draw event arrows (only for qualified signals)
            // -----------------------------------------------------------
            if (qualifiedBull)
                BullArrow[idx] = low - sc.TickSize * 4;
            if (qualifiedBear)
                BearArrow[idx] = high + sc.TickSize * 4;

            // -----------------------------------------------------------
            // 10. Zone rectangles for Wyckoff events
            //     Spring/SC → green zone, Upthrust/BC → purple zone
            //     Absorption bars also get zone rectangles
            //     Only qualified events get zone rectangles
            // -----------------------------------------------------------
            if (qualifiedBull)
            {
                if (showBullZones)
                {
                // Quality-scaled extension: high quality → long zone
                int scaledExt = zoneExt * bullQuality / 100;
                if (scaledExt < 3) scaledExt = 3;

                float zoneBottom = low - sc.TickSize;
                float zoneTop    = high + sc.TickSize;

                // Price invalidation: truncate if close breaks below zone
                int endBar = idx + scaledExt;
                if (endBar >= sc.ArraySize) endBar = sc.ArraySize - 1;
                for (int k = idx + 1; k <= endBar; k++)
                {
                    if (k < sc.ArraySize &&
                        sc.BaseData[SC_LAST][k] < zoneBottom)
                    {
                        endBar = k;
                        break;
                    }
                }

                s_UseTool tool;
                tool.Clear();
                tool.ChartNumber = sc.ChartNumber;
                tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
                tool.LineNumber = 70000 + idx;
                tool.AddMethod = UTAM_ADD_OR_ADJUST;
                tool.BeginIndex = idx;
                tool.EndIndex = endBar;
                tool.BeginValue = zoneBottom;
                tool.EndValue   = zoneTop;
                tool.Color = BullZoneRef.PrimaryColor;
                tool.SecondaryColor = tool.Color;
                tool.TransparencyLevel = zoneTrans;
                tool.AddAsUserDrawnDrawing = 0;
                sc.UseTool(tool);
                }
            }

            if (qualifiedBear)
            {
                if (showBearZones)
                {
                // Quality-scaled extension: high quality → long zone
                int scaledExt = zoneExt * bearQuality / 100;
                if (scaledExt < 3) scaledExt = 3;

                float zoneBottom = low - sc.TickSize;
                float zoneTop    = high + sc.TickSize;

                // Price invalidation: truncate if close breaks above zone
                int endBar = idx + scaledExt;
                if (endBar >= sc.ArraySize) endBar = sc.ArraySize - 1;
                for (int k = idx + 1; k <= endBar; k++)
                {
                    if (k < sc.ArraySize &&
                        sc.BaseData[SC_LAST][k] > zoneTop)
                    {
                        endBar = k;
                        break;
                    }
                }

                s_UseTool tool;
                tool.Clear();
                tool.ChartNumber = sc.ChartNumber;
                tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
                tool.LineNumber = 70000 + idx;
                tool.AddMethod = UTAM_ADD_OR_ADJUST;
                tool.BeginIndex = idx;
                tool.EndIndex = endBar;
                tool.BeginValue = zoneBottom;
                tool.EndValue   = zoneTop;
                tool.Color = BearZoneRef.PrimaryColor;
                tool.SecondaryColor = tool.Color;
                tool.TransparencyLevel = zoneTrans;
                tool.AddAsUserDrawnDrawing = 0;
                sc.UseTool(tool);
                }
            }

            // Absorption zone rectangles (smaller, quality-scaled)
            if (AbsFlag[idx] > 0.5f && !isSpring && !isSC &&
                !isUpthrust && !isBC)
            {
                if (showAbsZones)
                {
                // Absorption uses half the zone extension, quality-scaled
                int absBaseExt = zoneExt / 2;
                int absQ = (AbsType[idx] > 0.5f) ? bullQuality : bearQuality;
                if (absQ < 30) absQ = 30; // Absorption always gets min visibility
                int absScaledExt = absBaseExt * absQ / 100;
                if (absScaledExt < 2) absScaledExt = 2;

                // Body-based zone (tighter than wick-based)
                float bodyHi = (open > close) ? open : close;
                float bodyLo = (open < close) ? open : close;
                float absBottom = bodyLo - sc.TickSize;
                float absTop    = bodyHi + sc.TickSize;

                // Price invalidation by absorption direction
                int absEnd = idx + absScaledExt;
                if (absEnd >= sc.ArraySize) absEnd = sc.ArraySize - 1;
                bool isBullAbs = (AbsType[idx] > 0.5f);
                for (int k = idx + 1; k <= absEnd; k++)
                {
                    if (k < sc.ArraySize)
                    {
                        // Bull absorption invalidated by close below zone
                        // Bear absorption invalidated by close above zone
                        if (isBullAbs && sc.BaseData[SC_LAST][k] < absBottom)
                            { absEnd = k; break; }
                        if (!isBullAbs && sc.BaseData[SC_LAST][k] > absTop)
                            { absEnd = k; break; }
                    }
                }

                s_UseTool tool;
                tool.Clear();
                tool.ChartNumber = sc.ChartNumber;
                tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
                tool.LineNumber = 70000 + idx;
                tool.AddMethod = UTAM_ADD_OR_ADJUST;
                tool.BeginIndex = idx;
                tool.EndIndex = absEnd;
                tool.BeginValue = absBottom;
                tool.EndValue   = absTop;

                // Color by absorption type
                if (isBullAbs)
                    tool.Color = AbsZoneRef.PrimaryColor;
                else
                    tool.Color = AbsZoneRef.SecondaryColor;

                tool.SecondaryColor = tool.Color;
                tool.TransparencyLevel = zoneTrans;
                tool.AddAsUserDrawnDrawing = 0;
                sc.UseTool(tool);
                }
            }

        }  // end Pass 2 loop
    }
}
