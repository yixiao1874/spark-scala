package myscala.recommend

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating


object RecommendScala {
  def main(args: Array[String]) {

    // 加载并解析数据
    //val conf = new SparkConf().setAppName("RecommendScala").setMaster("spark://192.168.56.100:9000").setJars(List("out/artifacts/spark_scala_jar/spark-scala.jar"))
    val conf =new SparkConf().setAppName("RecommendScala").setMaster("spark://192.168.56.100:7077").
      set("spark.driver.host","192.168.56.100").setJars(List("out/artifacts/spark_scala_jar/spark-scala.jar"))
    /*conf.set("fs.defaultFS", "spark://192.168.56.100:8070")
    conf.set("mapreduce.job.jar", "target/hadoop.jar")*/
    val sc = new SparkContext(conf)
    val data = sc.textFile("hdfs://192.168.56.100:9000/input/test.csv")

    /**
      * Product ratings are on a scale of 1-5:
      * 5: Must see
      * 4: Will enjoy
      * 3: It's okay
      * 2: Fairly bad
      * 1: Awful
      */
    val ratings = data.map(_.split(',') match { case Array(user, product, rate) =>
      Rating(user.toInt, product.toInt, rate.toDouble)
    })

    //使用ALS训练数据建立推荐模型
    val rank = 10
    val numIterations = 20
    val model = ALS.train(ratings, rank, numIterations, 0.01)

    //从 ratings 中获得只包含用户和商品的数据集
    val usersProducts = ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }

    //使用推荐模型对用户商品进行预测评分，得到预测评分的数据集
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    //将真实评分数据集与预测评分数据集进行合并
    val ratesAndPreds = ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions).sortByKey()  //ascending or descending

    //然后计算均方差，注意这里没有调用 math.sqrt方法
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()

    //打印出均方差值
    println("Mean Squared Error = " + MSE)
    //Mean Squared Error = 1.37797097094789E-5


    val users=data.map(_.split(",") match {
      case Array(user, product, rate) => (user)
    }).distinct().collect()
    //users: Array[String] = Array(4, 2, 3, 1)

    users.foreach(
      user => {
        //依次为用户推荐商品
        var rs = model.recommendProducts(user.toInt, numIterations)
        var value = ""
        var key = 0

        //拼接推荐结果
        rs.foreach(r => {
          key = r.user
          value = value + r.product + ":" + r.rating + ","
        })

        println(key.toString+"   " + value)
      }
    )
  }
}
