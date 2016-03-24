
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.{axpy, scal}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom
import org.apache.spark.mllib.linalg.{Vector, Vectors}



class KMeans (
        var k: Int,
        var maxIterations: Int,
        var runs: Int,
        var initializationMode: String,
        var initializationSteps: Int,
        var epsilon: Double,
        var seed: Long) extends Serializable with Logging {

    /**
    * Constructs a KMeans instance with default parameters: {k: 2, maxIterations: 20, runs: 1,
    * initializationMode: "k-means||", initializationSteps: 5, epsilon: 1e-4, seed: random}.
    */
    def this() = this(2, 20, 1, KMeans.K_MEANS_PARALLEL, 5, 1e-4, Utils.random.nextLong())

    /**
    * Number of clusters to create (k).
    */
    def getK: Int = k

    /**
    * Set the number of clusters to create (k). Default: 2.
    */
    def setK(k: Int): this.type = {
        this.k = k
        this
    }

    /**
    * Maximum number of iterations allowed.
    */
    def getMaxIterations: Int = maxIterations

    /**
    * Set maximum number of iterations allowed. Default: 20.
    */
    def setMaxIterations(maxIterations: Int): this.type = {
        this.maxIterations = maxIterations
        this
    }

    /**
    * The initialization algorithm. This can be either "random" or "k-means||".
    */
    def getInitializationMode: String = initializationMode

    /**
    * Set the initialization algorithm. This can be either "random" to choose random points as
    * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
    * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
    */
    def setInitializationMode(initializationMode: String): this.type = {
        KMeans.validateInitMode(initializationMode)
        this.initializationMode = initializationMode
        this
    }


    // Internal version of setRuns for Python API, this should be removed at the same time as setRuns
    // this is done to avoid deprecation warnings in our build.
    mllib] def internalSetRuns(runs: Int): this.type = {
        if (runs <= 0) {
            throw new IllegalArgumentException("Number of runs must be positive")
        }
        if (runs != 1) {
            logWarning("Setting number of runs is deprecated and will have no effect in 2.0.0")
        }
        this.runs = runs
        this
    }

    /**
    * Number of steps for the k-means|| initialization mode
    */
    def getInitializationSteps: Int = initializationSteps

    /**
    * Set the number of steps for the k-means|| initialization mode. This is an advanced
    * setting -- the default of 5 is almost always enough. Default: 5.
    */
    def setInitializationSteps(initializationSteps: Int): this.type = {
        if (initializationSteps <= 0) {
            throw new IllegalArgumentException("Number of initialization steps must be positive")
        }
        this.initializationSteps = initializationSteps
        this
    }

    /**
    * The distance threshold within which we've consider centers to have converged.
    */
    def getEpsilon: Double = epsilon

    /**
    * Set the distance threshold within which we've consider centers to have converged.
    * If all centers move less than this Euclidean distance, we stop iterating one run.
    */
    def setEpsilon(epsilon: Double): this.type = {
        this.epsilon = epsilon
        this
    }

    /**
    * The random seed for cluster initialization.
    */
    def getSeed: Long = seed

    /**
    * Set the random seed for cluster initialization.
    */
    def setSeed(seed: Long): this.type = {
        this.seed = seed
        this
    }

    // Initial cluster centers can be provided as a KMeansModel object rather than using the
    // random or k-means|| initializationMode
    var initialModel: Option[KMeansModel] = None

    /**
    * Set the initial starting point, bypassing the random initialization or k-means||
    * The condition model.k == this.k must be met, failure results
    * in an IllegalArgumentException.
    */
    def setInitialModel(model: KMeansModel): this.type = {
        require(model.k == k, "mismatched cluster count")
        initialModel = Some(model)
        this
    }

    /**
    * Train a K-means model on the given set of points; `data` should be cached for high
    * performance, because this is an iterative algorithm.
    */
    def run(data: RDD[Vector]): KMeansModel = {

        if (data.getStorageLevel == StorageLevel.NONE) {
            logWarning("The input data is not directly cached, which may hurt performance if its"
                + " parent RDDs are also uncached.")
        }

        // Compute squared norms and cache them.
        val norms = data.map(Vectors.norm(_, 2.0))
        norms.persist()
        val zippedData = data.zip(norms).map { case (v, norm) =>
            new VectorWithNorm(v, norm)
        }
        val model = runAlgorithm(zippedData)
        norms.unpersist()

        // Warn at the end of the run as well, for increased visibility.
        if (data.getStorageLevel == StorageLevel.NONE) {
            logWarning("The input data was not directly cached, which may hurt performance if its"
                + " parent RDDs are also uncached.")
        }
        model
    }

    /**
    * Implementation of K-Means algorithm.
    */
    def runAlgorithm(data: RDD[VectorWithNorm]): KMeansModel = {
        println("TOTOTATATUTU")
        val sc = data.sparkContext

        val initStartTime = System.nanoTime()

        // Only one run is allowed when initialModel is given
        val numRuns = if (initialModel.nonEmpty) {
            if (runs > 1) logWarning("Ignoring runs; one run is allowed when initialModel is given.")
            1
        } else {
            runs
        }

        val centers = initialModel match {
            case Some(kMeansCenters) => {
                Array(kMeansCenters.clusterCenters.map(s => new VectorWithNorm(s)))
            }
            case None => {
                if (initializationMode == KMeans.RANDOM) {
                    initRandom(data)
                } else {
                    initKMeansParallel(data)
                }
            }
        }
        val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
        logInfo(s"Initialization with $initializationMode took " + "%.3f".format(initTimeInSeconds) +
        " seconds.")

        val active = Array.fill(numRuns)(true)
        val costs = Array.fill(numRuns)(0.0)

        var activeRuns = new ArrayBuffer[Int] ++ (0 until numRuns)
        var iteration = 0

        val iterationStartTime = System.nanoTime()

        // Execute iterations of Lloyd's algorithm until all runs have converged
        while (iteration < maxIterations && !activeRuns.isEmpty) {
            type WeightedPoint = (Vector, Long)
            def mergeContribs(x: WeightedPoint, y: WeightedPoint): WeightedPoint = {
                axpy(1.0, x._1, y._1)
                (y._1, x._2 + y._2)
            }

            val activeCenters = activeRuns.map(r => centers(r)).toArray
            val costAccums = activeRuns.map(_ => sc.accumulator(0.0))

            val bcActiveCenters = sc.broadcast(activeCenters)

            // Find the sum and count of points mapping to each center
            val totalContribs = data.mapPartitions { points =>
                val thisActiveCenters = bcActiveCenters.value
                val runs = thisActiveCenters.length
                val k = thisActiveCenters(0).length
                val dims = thisActiveCenters(0)(0).vector.size

                val sums = Array.fill(runs, k)(Vectors.zeros(dims))
                val counts = Array.fill(runs, k)(0L)

                points.foreach { point =>
                    (0 until runs).foreach { i =>
                        val (bestCenter, cost) = KMeans.findClosest(thisActiveCenters(i), point)
                        costAccums(i) += cost
                        val sum = sums(i)(bestCenter)
                        axpy(1.0, point.vector, sum)
                        counts(i)(bestCenter) += 1
                    }
                }

                val contribs = for (i <- 0 until runs; j <- 0 until k) yield {
                    ((i, j), (sums(i)(j), counts(i)(j)))
                }
                contribs.iterator
            }.reduceByKey(mergeContribs).collectAsMap()

            bcActiveCenters.unpersist(blocking = false)

            // Update the cluster centers and costs for each active run
            for ((run, i) <- activeRuns.zipWithIndex) {
                var changed = false
                var j = 0
                while (j < k) {
                    val (sum, count) = totalContribs((i, j))
                    if (count != 0) {
                        scal(1.0 / count, sum)
                        val newCenter = new VectorWithNorm(sum)
                        // if (KMeans.fastSquaredDistance(newCenter, centers(run)(j)) > epsilon * epsilon) {
                        if (KMeans.manhattanCircularDistance(newCenter, centers(run)(j)) > epsilon * epsilon) {
                            changed = true
                        }
                        centers(run)(j) = newCenter
                    }
                    j += 1
                }
                if (!changed) {
                    active(run) = false
                    logInfo("Run " + run + " finished in " + (iteration + 1) + " iterations")
                }
                costs(run) = costAccums(i).value
            }

            activeRuns = activeRuns.filter(active(_))
            iteration += 1
        }

        val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
        logInfo(s"Iterations took " + "%.3f".format(iterationTimeInSeconds) + " seconds.")

        if (iteration == maxIterations) {
            logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
        } else {
            logInfo(s"KMeans converged in $iteration iterations.")
        }

        val (minCost, bestRun) = costs.zipWithIndex.min

        logInfo(s"The cost for the best run is $minCost.")

        new KMeansModel(centers(bestRun).map(_.vector))
    }

    /**
    * Initialize `runs` sets of cluster centers at random.
    */
    def initRandom(data: RDD[VectorWithNorm])
    : Array[Array[VectorWithNorm]] = {
        // Sample all the cluster centers in one pass to avoid repeated scans
        val sample = data.takeSample(true, runs * k, new XORShiftRandom(this.seed).nextInt()).toSeq
        Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).map { v =>
            new VectorWithNorm(Vectors.dense(v.vector.toArray), v.norm)
        }.toArray)
    }

    /**
    * Initialize `runs` sets of cluster centers using the k-means|| algorithm by Bahmani et al.
    * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
    * to find with dissimilar cluster centers by starting with a random center and then doing
    * passes where more centers are chosen with probability proportional to their squared distance
    * to the current cluster set. It results in a provable approximation to an optimal clustering.
    *
    * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
    */
    def initKMeansParallel(data: RDD[VectorWithNorm])
    : Array[Array[VectorWithNorm]] = {
        // Initialize empty centers and point costs.
        val centers = Array.tabulate(runs)(r => ArrayBuffer.empty[VectorWithNorm])
        var costs = data.map(_ => Array.fill(runs)(Double.PositiveInfinity))

        // Initialize each run's first center to a random point.
        val seed = new XORShiftRandom(this.seed).nextInt()
        val sample = data.takeSample(true, runs, seed).toSeq
        val newCenters = Array.tabulate(runs)(r => ArrayBuffer(sample(r).toDense))

        /** Merges new centers to centers. */
        def mergeNewCenters(): Unit = {
            var r = 0
            while (r < runs) {
                centers(r) ++= newCenters(r)
                newCenters(r).clear()
                r += 1
            }
        }

        // On each step, sample 2 * k points on average for each run with probability proportional
        // to their squared distance from that run's centers. Note that only distances between points
        // and new centers are computed in each iteration.
        var step = 0
        while (step < initializationSteps) {
            val bcNewCenters = data.context.broadcast(newCenters)
            val preCosts = costs
            costs = data.zip(preCosts).map { case (point, cost) =>
                Array.tabulate(runs) { r =>
                    math.min(KMeans.pointCost(bcNewCenters.value(r), point), cost(r))
                }
            }.persist(StorageLevel.MEMORY_AND_DISK)
            val sumCosts = costs
            .aggregate(new Array[Double](runs))(
                seqOp = (s, v) => {
                    // s += v
                    var r = 0
                    while (r < runs) {
                        s(r) += v(r)
                        r += 1
                    }
                    s
                },
                combOp = (s0, s1) => {
                    // s0 += s1
                    var r = 0
                    while (r < runs) {
                        s0(r) += s1(r)
                        r += 1
                    }
                    s0
                }
            )

            bcNewCenters.unpersist(blocking = false)
            preCosts.unpersist(blocking = false)

            val chosen = data.zip(costs).mapPartitionsWithIndex { (index, pointsWithCosts) =>
                val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
                pointsWithCosts.flatMap { case (p, c) =>
                    val rs = (0 until runs).filter { r =>
                        rand.nextDouble() < 2.0 * c(r) * k / sumCosts(r)
                    }
                    if (rs.length > 0) Some((p, rs)) else None
                }
            }.collect()
            mergeNewCenters()
            chosen.foreach { case (p, rs) =>
                rs.foreach(newCenters(_) += p.toDense)
            }
            step += 1
        }

        mergeNewCenters()
        costs.unpersist(blocking = false)

        // Finally, we might have a set of more than k candidate centers for each run; weigh each
        // candidate by the number of points in the dataset mapping to it and run a local k-means++
        // on the weighted centers to pick just k of them
        val bcCenters = data.context.broadcast(centers)
        val weightMap = data.flatMap { p =>
            Iterator.tabulate(runs) { r =>
                ((r, KMeans.findClosest(bcCenters.value(r), p)._1), 1.0)
            }
        }.reduceByKey(_ + _).collectAsMap()

        bcCenters.unpersist(blocking = false)

        val finalCenters = (0 until runs).par.map { r =>
            val myCenters = centers(r).toArray
            val myWeights = (0 until myCenters.length).map(i => weightMap.getOrElse((r, i), 0.0)).toArray
            LocalKMeans.kMeansPlusPlus(r, myCenters, myWeights, k, 30)
        }

        finalCenters.toArray
    }
}

object KMeans {

    // Initialization mode names
    val RANDOM = "random"
    val K_MEANS_PARALLEL = "k-means||"

    /**
    * Trains a k-means model using the given set of parameters.
    *
    * @param data Training points as an `RDD` of `Vector` types.
    * @param k Number of clusters to create.
    * @param maxIterations Maximum number of iterations allowed.
    * @param runs Number of runs to execute in parallel. The best model according to the cost
    *             function will be returned. (default: 1)
    * @param initializationMode The initialization algorithm. This can either be "random" or
    *                           "k-means||". (default: "k-means||")
    * @param seed Random seed for cluster initialization. Default is to generate seed based
    *             on system time.
    */
    def train(
            data: RDD[Vector],
            k: Int,
            maxIterations: Int,
            runs: Int,
            initializationMode: String,
            seed: Long): KMeansModel = {
        new KMeans().setK(k)
        .setMaxIterations(maxIterations)
        .internalSetRuns(runs)
        .setInitializationMode(initializationMode)
        .setSeed(seed)
        .run(data)
    }

    /**
    * Trains a k-means model using the given set of parameters.
    *
    * @param data Training points as an `RDD` of `Vector` types.
    * @param k Number of clusters to create.
    * @param maxIterations Maximum number of iterations allowed.
    * @param runs Number of runs to execute in parallel. The best model according to the cost
    *             function will be returned. (default: 1)
    * @param initializationMode The initialization algorithm. This can either be "random" or
    *                           "k-means||". (default: "k-means||")
    */
    def train(
            data: RDD[Vector],
            k: Int,
            maxIterations: Int,
            runs: Int,
            initializationMode: String): KMeansModel = {
        new KMeans().setK(k)
        .setMaxIterations(maxIterations)
        .internalSetRuns(runs)
        .setInitializationMode(initializationMode)
        .run(data)
    }

    /**
    * Trains a k-means model using specified parameters and the default values for unspecified.
    */
    def train(
        data: RDD[Vector],
        k: Int,
        maxIterations: Int): KMeansModel = {
            train(data, k, maxIterations, 1, K_MEANS_PARALLEL)
    }

    /**
    * Trains a k-means model using specified parameters and the default values for unspecified.
    */
    def train(
            data: RDD[Vector],
            k: Int,
            maxIterations: Int,
            runs: Int): KMeansModel = {
        train(data, k, maxIterations, runs, K_MEANS_PARALLEL)
    }

    /**
    * Returns the index of the closest center to the given point, as well as the squared distance.
    */
    def findClosest(
            centers: TraversableOnce[VectorWithNorm],
            point: VectorWithNorm): (Int, Double) = {
        var bestDistance = Double.PositiveInfinity
        var bestIndex = 0
        var i = 0
        centers.foreach { center =>
            // distance computation.
            var lowerBoundOfSqDist = center.norm - point.norm
            lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
            if (lowerBoundOfSqDist < bestDistance) {
                // val distance: Double = fastSquaredDistance(center, point)
                val distance: Double = manhattanCircularDistance(center, point)
                if (distance < bestDistance) {
                    bestDistance = distance
                    bestIndex = i
                }
            }
            i += 1
        }
        (bestIndex, bestDistance)
    }

    /**
    * Returns the K-means cost of a given point against the given cluster centers.
    */
    def pointCost(
            centers: TraversableOnce[VectorWithNorm],
            point: VectorWithNorm): Double =
        findClosest(centers, point)._2

    /**
    * Returns the squared Euclidean distance between two vectors computed by
    * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
    */
    def fastSquaredDistance(
            v1: VectorWithNorm,
            v2: VectorWithNorm): Double = {
        org.apache.spark.mllib.util.MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
    }

    def manhattanDistance(
            v1: VectorWithNorm,
            v2: VectorWithNorm): Double = {
        var sum: Double = 0
        for (i <- 0 until v1.vector.size) {
            sum += scala.math.abs(v2.vector(i)-v1.vector(i)
        }
        sum
    }

    def euclideanCircularDistance(
            v1: VectorWithNorm,
            v2: VectorWithNorm): Double = {
        var sum: Double = 0
        var total = v1.vector.size
        for (i <- 0 until (total-2)) {
            sum += scala.math.pow((v2.vector(i)-v1.vector(i)),2)
        }
        for (i <- (total-2) until total) {
            sum += scala.math.pow(circularDistance(v2.vector(i),v1.vector(i),7.0),2)
        }
        scala.math.sqrt(sum)
    }

    def euclideanDissimilarityDistance(
            v1: VectorWithNorm,
            v2: VectorWithNorm): Double = {
        var sum: Double = 0.0
        var total = v1.vector.size
        for (i <- 0 until (total-2)) {
            sum += scala.math.pow((v2.vector(i)-v1.vector(i)),2)
        }
        for (i <- (total-2) until total) {
          var increment :Double = if ((v2.vector(i) - v1.vector(i)) >  1) 1 else 0
            sum += increment
        }
        scala.math.sqrt(sum)
    }

    def manhattanCircularDistance(
            v1: VectorWithNorm,
            v2: VectorWithNorm): Double = {
        var sum: Double = 0
        var total = v1.vector.size
        for (i <- 0 until (total-2)) {
            sum += scala.math.abs((v2.vector(i)-v1.vector(i)))
        }
        for (i <- (total-2) until total) {
            sum += scala.math.abs(circularDistance(v2.vector(i),v1.vector(i),7.0))
        }
        sum
    }

    def manhattanDissimilarityDistance(
            v1: VectorWithNorm,
            v2: VectorWithNorm): Double = {
        var sum: Double = 0.0
        var total = v1.vector.size
        for (i <- 0 until (total-2)) {
            sum += scala.math.abs((v2.vector(i)-v1.vector(i)))
        }
        for (i <- (total-2) until total) {
          var increment :Double = if ((v2.vector(i) - v1.vector(i)) >  1) 1 else 0
            sum += increment
        }
        sum
    }

    def circularDistance(
            v1: Double,
            v2: Double,
            m: Double): Double = {
        scala.math.sqrt(
            1-scala.math.abs(
                1-(2*scala.math.abs((v2-v1)/m))
            )
        )
    }

    def validateInitMode(initMode: String): Boolean = {
        initMode match {
            case KMeans.RANDOM => true
            case KMeans.K_MEANS_PARALLEL => true
            case _ => false
        }
    }
}


class VectorWithNorm(val vector:  org.apache.spark.mllib.linalg.Vector, val norm: Double) extends Serializable {
    def this(vector:  org.apache.spark.mllib.linalg.Vector) = this(vector, org.apache.spark.mllib.linalg.Vectors.norm(vector, 2.0))
    def this(array: Array[Double]) = this(org.apache.spark.mllib.linalg.Vectors.dense(array))
    def toDense: VectorWithNorm = new VectorWithNorm(org.apache.spark.mllib.linalg.Vectors.dense(vector.toArray), norm)
}
