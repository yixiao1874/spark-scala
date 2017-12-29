package myscala.count

import org.apache.spark.{SparkContext, SparkConf}

object WordCount {
  def main(args : Array[String]) {
    val conf = new SparkConf().setAppName("WordCount")
    val sc = new SparkContext(conf)
    val input = "F:\\Hadoop\\test\\input\\test.txt"
    val texts = sc.textFile(input).map(line => line.split(" "))
      .flatMap(words => words.map(word => (word.replaceAll("[^A-Za-z]", ""), 1)))
    val counts =  texts.reduceByKey(_ + _)
    counts.collect.foreach{
      case (word, num) =>
        println(word + " " + num.toString)
    }

  }
}