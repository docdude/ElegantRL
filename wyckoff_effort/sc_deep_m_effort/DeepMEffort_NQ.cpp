// ==========================================================================
// Deep-M Effort NQ — Sierra Chart Advanced Custom Study (ACSIL)
// File: DeepMEffort_NQ.cpp
//
// Chart Setup: NQ futures, Range 40 bar type
// Compile: Analysis > Build Custom Studies DLL > Remote Build
//
// This study implements:
//   - Effort Index composite (volume/delta/speed ratios)
//   - Absorption zone detection (high volume, weak delta, slow bars)
//   - Vacuum zone detection (low volume, fast bars)
//   - Green zones = bullish pressure (path of least resistance UP)
//   - Purple zones = bearish pressure (path of least resistance DOWN)
//   - Price EMA with slope-based coloring
// ==========================================================================

#include "sierrachart.h"
#include <cmath>

SCDLLName("DeepMEffort_NQ")

// -------------------------------------------------------------------------
// Helper: Exponential Moving Average (manual, for use in manual loop)
// -------------------------------------------------------------------------
static float CalcEMA(float currentValue, float prevEMA, int period)
{
    float k = 2.0f / (period + 1.0f);
    return currentValue * k + prevEMA * (1.0f - k);
}

// =========================================================================
// Main Study Function
// =========================================================================
SCSFExport scsf_DeepMEffort_NQ(SCStudyInterfaceRef sc)
{
    // --- Subgraphs (Region 0 = price chart overlay) ---
    SCSubgraphRef PriceEMA       = sc.Subgraph[0];   // Price EMA line
    SCSubgraphRef BullSignal     = sc.Subgraph[1];   // Bullish zone signal triangles
    SCSubgraphRef BearSignal     = sc.Subgraph[2];   // Bearish zone signal triangles

    // --- Subgraphs (internal computation, not displayed) ---
    SCSubgraphRef EffortIndex   = sc.Subgraph[3];
    SCSubgraphRef EffortEMA     = sc.Subgraph[4];
    SCSubgraphRef EffortUpper   = sc.Subgraph[5];
    SCSubgraphRef EffortLower   = sc.Subgraph[6];
    SCSubgraphRef AbsorptionSc  = sc.Subgraph[7];

    // --- Extra arrays for intermediate calculations (all on PriceEMA) ---
    // PriceEMA.Arrays[0] = Volume EMA
    // PriceEMA.Arrays[1] = Volume Ratio
    // PriceEMA.Arrays[2] = Delta Pct
    // PriceEMA.Arrays[3] = Delta Pct EMA
    // PriceEMA.Arrays[4] = Speed
    // PriceEMA.Arrays[5] = Speed EMA
    // PriceEMA.Arrays[6] = Speed Ratio
    // PriceEMA.Arrays[7] = Delta Divergence
    // NOTE: Signal subgraphs (BullSignal, BearSignal) must have NO extra
    // array usage — keep them clean for SC's arrow rendering engine.

    SCFloatArrayRef VolEMA       = PriceEMA.Arrays[0];
    SCFloatArrayRef VolRatio     = PriceEMA.Arrays[1];
    SCFloatArrayRef DeltaPct     = PriceEMA.Arrays[2];
    SCFloatArrayRef DeltaPctEMA  = PriceEMA.Arrays[3];
    SCFloatArrayRef Speed        = PriceEMA.Arrays[4];
    SCFloatArrayRef SpeedEMA     = PriceEMA.Arrays[5];
    SCFloatArrayRef SpeedRatio   = PriceEMA.Arrays[6];
    SCFloatArrayRef DeltaDiv     = PriceEMA.Arrays[7];

    // --- Inputs ---
    SCInputRef InEMAPeriod           = sc.Input[0];
    SCInputRef InZoneStdMult         = sc.Input[1];
    SCInputRef InAbsorptionThreshold = sc.Input[2];
    SCInputRef InVacuumThreshold     = sc.Input[3];
    SCInputRef InDeltaEMAPeriod      = sc.Input[4];
    SCInputRef InTimeEMAPeriod       = sc.Input[5];
    SCInputRef InZoneExtension       = sc.Input[6];
    SCInputRef InNormPeriod          = sc.Input[7];
    SCInputRef InDeltaFilter         = sc.Input[8];
    SCInputRef InSpeedFilter         = sc.Input[9];
    SCInputRef InAbsContinuation     = sc.Input[10];
    SCInputRef InVacContinuation     = sc.Input[11];
    SCInputRef InMinZoneBars         = sc.Input[12];

    // =====================================================================
    // DEFAULTS
    // =====================================================================
    if (sc.SetDefaults)
    {
        sc.GraphName = "Deep-M Effort NQ";
        sc.AutoLoop = 0;   // Manual looping for zone detection
        sc.GraphRegion = 0; // Overlay on price chart
        sc.ScaleRangeType = SCALE_SAMEASREGION; // Use same scale as price chart

        // --- Price EMA (overlay) ---
        PriceEMA.Name = "Price EMA";
        PriceEMA.DrawStyle = DRAWSTYLE_LINE;
        PriceEMA.PrimaryColor = RGB(76, 175, 80);     // Green (above)
        PriceEMA.SecondaryColor = RGB(156, 39, 176);   // Purple (below)
        PriceEMA.SecondaryColorUsed = 1;
        PriceEMA.LineWidth = 2;

        // --- Signal triangles on price chart ---
        BullSignal.Name = "Bull Arrow";
        BullSignal.DrawStyle = DRAWSTYLE_ARROW_UP;
        BullSignal.PrimaryColor = RGB(0, 230, 118);    // Bright green (#00E676)
        BullSignal.LineWidth = 4;
        BullSignal.DrawZeros = 0;

        BearSignal.Name = "Bear Arrow";
        BearSignal.DrawStyle = DRAWSTYLE_ARROW_DOWN;
        BearSignal.PrimaryColor = RGB(255, 23, 68);    // Bright red (#FF1744)
        BearSignal.LineWidth = 4;
        BearSignal.DrawZeros = 0;

        // --- Effort study ---
        // These subgraphs compute internally but are hidden from display
        // because SC v2889 subgraphs can't be assigned to separate regions.
        // The zone rectangles are drawn via s_UseTool on the price chart.
        // To see effort values, use the study's Spreadsheet output.
        EffortIndex.Name = "Effort Index";
        EffortIndex.DrawStyle = DRAWSTYLE_IGNORE;

        EffortEMA.Name = "Effort EMA";
        EffortEMA.DrawStyle = DRAWSTYLE_IGNORE;

        EffortUpper.Name = "Effort Upper Band";
        EffortUpper.DrawStyle = DRAWSTYLE_IGNORE;

        EffortLower.Name = "Effort Lower Band";
        EffortLower.DrawStyle = DRAWSTYLE_IGNORE;

        AbsorptionSc.Name = "Absorption Score";
        AbsorptionSc.DrawStyle = DRAWSTYLE_IGNORE;

        // --- Inputs ---
        InEMAPeriod.Name = "EMA Period";
        InEMAPeriod.SetInt(20);
        InEMAPeriod.SetIntLimits(5, 200);

        InZoneStdMult.Name = "Zone Std Multiplier";
        InZoneStdMult.SetFloat(1.5f);
        InZoneStdMult.SetFloatLimits(0.5f, 4.0f);

        InAbsorptionThreshold.Name = "Absorption Threshold (Vol/EMA)";
        InAbsorptionThreshold.SetFloat(1.5f);
        InAbsorptionThreshold.SetFloatLimits(1.0f, 5.0f);

        InVacuumThreshold.Name = "Vacuum Threshold (Vol/EMA)";
        InVacuumThreshold.SetFloat(0.6f);
        InVacuumThreshold.SetFloatLimits(0.1f, 1.5f);

        InDeltaEMAPeriod.Name = "Delta EMA Period";
        InDeltaEMAPeriod.SetInt(14);
        InDeltaEMAPeriod.SetIntLimits(3, 100);

        InTimeEMAPeriod.Name = "Time/Speed EMA Period";
        InTimeEMAPeriod.SetInt(14);
        InTimeEMAPeriod.SetIntLimits(3, 100);

        InZoneExtension.Name = "Zone Extension (bars forward)";
        InZoneExtension.SetInt(15);
        InZoneExtension.SetIntLimits(5, 200);

        InNormPeriod.Name = "Normalization Lookback";
        InNormPeriod.SetInt(100);
        InNormPeriod.SetIntLimits(20, 500);

        InDeltaFilter.Name = "Delta Filter (max |delta%|)";
        InDeltaFilter.SetFloat(0.20f);
        InDeltaFilter.SetFloatLimits(0.05f, 0.50f);

        InSpeedFilter.Name = "Vacuum Speed Filter (min SR)";
        InSpeedFilter.SetFloat(1.1f);
        InSpeedFilter.SetFloatLimits(0.5f, 3.0f);

        InAbsContinuation.Name = "Absorption Continuation Factor";
        InAbsContinuation.SetFloat(0.6f);
        InAbsContinuation.SetFloatLimits(0.3f, 1.0f);

        InVacContinuation.Name = "Vacuum Continuation Factor";
        InVacContinuation.SetFloat(1.3f);
        InVacContinuation.SetFloatLimits(1.0f, 2.0f);

        InMinZoneBars.Name = "Min Zone Bars";
        InMinZoneBars.SetInt(2);
        InMinZoneBars.SetIntLimits(1, 10);

        return;
    }

    // =====================================================================
    // MANUAL LOOP — Compute all bars
    // =====================================================================
    int emaPeriod   = InEMAPeriod.GetInt();
    int deltaEMA    = InDeltaEMAPeriod.GetInt();
    int timeEMA     = InTimeEMAPeriod.GetInt();
    int normPeriod  = InNormPeriod.GetInt();
    float stdMult   = InZoneStdMult.GetFloat();
    float absThresh = InAbsorptionThreshold.GetFloat();
    float vacThresh = InVacuumThreshold.GetFloat();
    int zoneExt     = InZoneExtension.GetInt();
    float deltaFilter   = InDeltaFilter.GetFloat();
    float speedFilter   = InSpeedFilter.GetFloat();
    float absCont       = InAbsContinuation.GetFloat();
    float vacCont       = InVacContinuation.GetFloat();
    int minZoneBars     = InMinZoneBars.GetInt();
    int maxCluster  = 40;  // max cluster bars for lookback calculation

    // =====================================================================
    // PASS 1: Compute all metrics for every bar
    // =====================================================================
    for (int idx = sc.UpdateStartIndex; idx < sc.ArraySize; ++idx)
    {
        // ---- Access bar data via sc.BaseData ----
        float open   = sc.BaseData[SC_OPEN][idx];
        float high   = sc.BaseData[SC_HIGH][idx];
        float low    = sc.BaseData[SC_LOW][idx];
        float close  = sc.BaseData[SC_LAST][idx];
        float volume = sc.BaseData[SC_VOLUME][idx];
        float askVol = sc.BaseData[SC_ASKVOL][idx];
        float bidVol = sc.BaseData[SC_BIDVOL][idx];
        float numTrades = sc.BaseData[SC_NUM_TRADES][idx];

        float delta = askVol - bidVol;
        float deltaPct = (volume > 0) ? delta / volume : 0.0f;

        // Zero signal arrays for every bar (required for AutoLoop=0).
        // Pass 2 will override with non-zero values at zone start bars.
        BullSignal[idx] = 0;
        BearSignal[idx] = 0;
        DeltaPct[idx] = deltaPct;

        // ---- Volume EMA & Ratio ----
        if (idx < emaPeriod)
        {
            float sum = 0;
            for (int k = 0; k <= idx; k++)
                sum += sc.BaseData[SC_VOLUME][k];
            VolEMA[idx] = sum / (idx + 1);
        }
        else if (idx == emaPeriod)
        {
            float sum = 0;
            for (int k = 0; k < emaPeriod; k++)
                sum += sc.BaseData[SC_VOLUME][k];
            VolEMA[idx] = CalcEMA(volume, sum / emaPeriod, emaPeriod);
        }
        else
        {
            VolEMA[idx] = CalcEMA(volume, VolEMA[idx - 1], emaPeriod);
        }
        VolRatio[idx] = (VolEMA[idx] > 0) ? volume / VolEMA[idx] : 1.0f;

        // ---- Delta % EMA ----
        if (idx == 0)
            DeltaPctEMA[idx] = deltaPct;
        else
            DeltaPctEMA[idx] = CalcEMA(deltaPct, DeltaPctEMA[idx - 1], deltaEMA);

        // ---- Speed ----
        float speed = (numTrades > 1) ? volume / numTrades : 1.0f;
        Speed[idx] = speed;
        if (idx == 0)
            SpeedEMA[idx] = speed;
        else
            SpeedEMA[idx] = CalcEMA(speed, SpeedEMA[idx - 1], timeEMA);
        SpeedRatio[idx] = (SpeedEMA[idx] > 0) ? speed / SpeedEMA[idx] : 1.0f;

        // ---- Delta Divergence ----
        float absDeltaPct = fabsf(deltaPct);
        DeltaDiv[idx] = (absDeltaPct > 0.001f)
            ? VolRatio[idx] / (absDeltaPct + 0.01f) : 0.0f;

        // ---- Rolling normalization for Effort Index ----
        float vrMin = VolRatio[idx], vrMax = VolRatio[idx];
        float ddMin = DeltaDiv[idx], ddMax = DeltaDiv[idx];
        float srMin = SpeedRatio[idx], srMax = SpeedRatio[idx];
        int lookStart = (idx - normPeriod + 1 > 0) ? idx - normPeriod + 1 : 0;
        for (int k = lookStart; k <= idx; k++)
        {
            if (VolRatio[k] < vrMin) vrMin = VolRatio[k];
            if (VolRatio[k] > vrMax) vrMax = VolRatio[k];
            if (DeltaDiv[k] < ddMin) ddMin = DeltaDiv[k];
            if (DeltaDiv[k] > ddMax) ddMax = DeltaDiv[k];
            if (SpeedRatio[k] < srMin) srMin = SpeedRatio[k];
            if (SpeedRatio[k] > srMax) srMax = SpeedRatio[k];
        }

        float vrNorm = (vrMax - vrMin > 1e-6f)
            ? (VolRatio[idx] - vrMin) / (vrMax - vrMin) : 0.5f;
        float ddNorm = (ddMax - ddMin > 1e-6f)
            ? (DeltaDiv[idx] - ddMin) / (ddMax - ddMin) : 0.5f;
        float srNorm = (srMax - srMin > 1e-6f)
            ? (SpeedRatio[idx] - srMin) / (srMax - srMin) : 0.5f;

        float effortRaw = 0.50f * vrNorm + 0.25f * ddNorm + 0.25f * srNorm;
        EffortIndex[idx] = effortRaw;

        // ---- Effort EMA ----
        if (idx == 0)
            EffortEMA[idx] = effortRaw;
        else
            EffortEMA[idx] = CalcEMA(effortRaw, EffortEMA[idx - 1], emaPeriod);

        // ---- Rolling Std for bands ----
        if (idx >= emaPeriod - 1)
        {
            float sumVal = 0, sumSq = 0;
            for (int k = idx - emaPeriod + 1; k <= idx; k++)
            {
                float v = EffortIndex[k];
                sumVal += v;
                sumSq += v * v;
            }
            float mean = sumVal / emaPeriod;
            float variance = sumSq / emaPeriod - mean * mean;
            float stdDev = (variance > 0) ? sqrtf(variance) : 0.0f;
            EffortUpper[idx] = EffortEMA[idx] + stdMult * stdDev;
            EffortLower[idx] = EffortEMA[idx] - stdMult * stdDev;
        }

        // ---- Absorption Score ----
        float absScore = VolRatio[idx] * (1.0f - fabsf(deltaPct));
        if (SpeedRatio[idx] > 0)
            absScore /= (SpeedRatio[idx] + 0.1f);
        AbsorptionSc[idx] = absScore;

        // ---- Price EMA with slope coloring ----
        if (idx == 0)
            PriceEMA[idx] = close;
        else
            PriceEMA[idx] = CalcEMA(close, PriceEMA[idx - 1], emaPeriod);

        if (close >= PriceEMA[idx])
            PriceEMA.DataColor[idx] = PriceEMA.PrimaryColor;
        else
            PriceEMA.DataColor[idx] = PriceEMA.SecondaryColor;
    }  // end pass 1

    // =====================================================================
    // PASS 2: Zone Detection
    // Runs on every update. On full recalc (UpdateStartIndex==0), scans
    // all bars. On live updates, re-scans a lookback window so new zones
    // appear in real-time as bars complete.
    // =====================================================================
    {
        // Determine zone scan range
        int zoneScanStart;
        if (sc.UpdateStartIndex == 0)
        {
            // Full recalc: clear all zones and scan everything
            for (int ln = 50000; ln < 50000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
            for (int ln = 60000; ln < 60000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
            zoneScanStart = emaPeriod;
        }
        else
        {
            // Live update: re-scan recent window (lookback enough to
            // catch clusters forming). Delete stale zone drawings in
            // this window before redrawing.
            int lookback = maxCluster + zoneExt + 5;
            zoneScanStart = sc.UpdateStartIndex - lookback;
            if (zoneScanStart < emaPeriod)
                zoneScanStart = emaPeriod;

            for (int ln = 50000 + zoneScanStart; ln < 50000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
            for (int ln = 60000 + zoneScanStart; ln < 60000 + sc.ArraySize; ln++)
                sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_CHARTDRAWING, ln);
        }

        for (int idx = zoneScanStart; idx < sc.ArraySize; ++idx)
        {
            float deltaPct = DeltaPct[idx];

            // Absorption zone: high vol ratio + weak delta
            if (VolRatio[idx] >= absThresh && fabsf(deltaPct) < deltaFilter)
            {
                int clusterEnd = idx;
                for (int j = idx + 1; j < sc.ArraySize; j++)
                {
                    if (VolRatio[j] >= absThresh * absCont)
                        clusterEnd = j;
                    else
                        break;
                }

                if (clusterEnd - idx + 1 >= minZoneBars)
                {
                    // Use bar bodies (open/close) not wicks for tighter zones
                    float bodyHi = sc.BaseData[SC_OPEN][idx] > sc.BaseData[SC_LAST][idx]
                        ? sc.BaseData[SC_OPEN][idx] : sc.BaseData[SC_LAST][idx];
                    float bodyLo = sc.BaseData[SC_OPEN][idx] < sc.BaseData[SC_LAST][idx]
                        ? sc.BaseData[SC_OPEN][idx] : sc.BaseData[SC_LAST][idx];
                    float zoneHigh = bodyHi;
                    float zoneLow  = bodyLo;
                    float dirSum = 0;
                    for (int j = idx; j <= clusterEnd; j++)
                    {
                        float bh = sc.BaseData[SC_OPEN][j] > sc.BaseData[SC_LAST][j]
                            ? sc.BaseData[SC_OPEN][j] : sc.BaseData[SC_LAST][j];
                        float bl = sc.BaseData[SC_OPEN][j] < sc.BaseData[SC_LAST][j]
                            ? sc.BaseData[SC_OPEN][j] : sc.BaseData[SC_LAST][j];
                        if (bh > zoneHigh) zoneHigh = bh;
                        if (bl < zoneLow)  zoneLow = bl;
                        dirSum += (sc.BaseData[SC_LAST][j] >= sc.BaseData[SC_OPEN][j])
                                  ? 1.0f : -1.0f;
                    }
                    zoneHigh += sc.TickSize;
                    zoneLow  -= sc.TickSize;

                    bool isBullish = (dirSum < 0);

                    s_UseTool tool;
                    tool.Clear();
                    tool.ChartNumber = sc.ChartNumber;
                    tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
                    tool.LineNumber = 50000 + idx;
                    tool.AddMethod = UTAM_ADD_OR_ADJUST;
                    tool.BeginIndex = idx;
                    tool.EndIndex = clusterEnd + zoneExt;
                    tool.BeginValue = zoneLow;
                    tool.EndValue = zoneHigh;
                    tool.Color = isBullish
                        ? RGB(76, 175, 80)
                        : RGB(156, 39, 176);
                    tool.SecondaryColor = tool.Color;
                    tool.TransparencyLevel = 80;
                    tool.AddAsUserDrawnDrawing = 0;
                    sc.UseTool(tool);

                    // Arrow marker at zone start bar
                    if (isBullish)
                        BullSignal[idx] = zoneLow - sc.TickSize * 4;
                    else
                        BearSignal[idx] = zoneHigh + sc.TickSize * 4;

                    idx = clusterEnd;
                }
            }

            // Vacuum zone: low vol ratio + fast speed
            if (VolRatio[idx] <= vacThresh && SpeedRatio[idx] > speedFilter)
            {
                int clusterEnd = idx;
                for (int j = idx + 1; j < sc.ArraySize; j++)
                {
                    if (VolRatio[j] <= vacThresh * vacCont)
                        clusterEnd = j;
                    else
                        break;
                }

                if (clusterEnd - idx + 1 >= minZoneBars)
                {
                    // Use bar bodies (open/close) not wicks for tighter zones
                    float bodyHi = sc.BaseData[SC_OPEN][idx] > sc.BaseData[SC_LAST][idx]
                        ? sc.BaseData[SC_OPEN][idx] : sc.BaseData[SC_LAST][idx];
                    float bodyLo = sc.BaseData[SC_OPEN][idx] < sc.BaseData[SC_LAST][idx]
                        ? sc.BaseData[SC_OPEN][idx] : sc.BaseData[SC_LAST][idx];
                    float zoneHigh = bodyHi;
                    float zoneLow  = bodyLo;
                    float dirSum = 0;
                    for (int j = idx; j <= clusterEnd; j++)
                    {
                        float bh = sc.BaseData[SC_OPEN][j] > sc.BaseData[SC_LAST][j]
                            ? sc.BaseData[SC_OPEN][j] : sc.BaseData[SC_LAST][j];
                        float bl = sc.BaseData[SC_OPEN][j] < sc.BaseData[SC_LAST][j]
                            ? sc.BaseData[SC_OPEN][j] : sc.BaseData[SC_LAST][j];
                        if (bh > zoneHigh) zoneHigh = bh;
                        if (bl < zoneLow)  zoneLow = bl;
                        dirSum += (sc.BaseData[SC_LAST][j] >= sc.BaseData[SC_OPEN][j])
                                  ? 1.0f : -1.0f;
                    }
                    zoneHigh += sc.TickSize;
                    zoneLow  -= sc.TickSize;

                    bool isBullish = (dirSum > 0);

                    s_UseTool tool;
                    tool.Clear();
                    tool.ChartNumber = sc.ChartNumber;
                    tool.DrawingType = DRAWING_RECTANGLEHIGHLIGHT;
                    tool.LineNumber = 60000 + idx;
                    tool.AddMethod = UTAM_ADD_OR_ADJUST;
                    tool.BeginIndex = idx;
                    tool.EndIndex = clusterEnd + zoneExt;
                    tool.BeginValue = zoneLow;
                    tool.EndValue = zoneHigh;
                    tool.Color = isBullish
                        ? RGB(76, 175, 80)
                        : RGB(156, 39, 176);
                    tool.SecondaryColor = tool.Color;
                    tool.TransparencyLevel = 85;
                    tool.AddAsUserDrawnDrawing = 0;
                    sc.UseTool(tool);

                    // Arrow marker at zone start bar
                    if (isBullish)
                        BullSignal[idx] = zoneLow - sc.TickSize * 4;
                    else
                        BearSignal[idx] = zoneHigh + sc.TickSize * 4;

                    idx = clusterEnd;
                }
            }
        }  // end zone loop
    }  // end pass 2
}
