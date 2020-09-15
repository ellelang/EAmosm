def isSegmentOutlier(segKpiValue: SegKpiValue, segCount: SegCount, avgKpiValue: SegKpiValue, stdKpiValue: SegKpiValue): Boolean = {
    val probability = 1.0 / (4 * segCount)
    val standardNormal = new NormalDistribution()
    val maximumAllowableDeviation = math.abs(standardNormal.inverseCumulativeProbability(probability))
    val deviation = math.abs(segKpiValue - avgKpiValue) / stdKpiValue

    maximumAllowableDeviation < deviation
  }
