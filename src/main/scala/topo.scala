//2018.06.20更新：在原来热点聚类的基础上新增拓扑图功能；另外，信件标题通过另一个python文件生成，所以，聚类信件的时间限制在标题生成的python脚本中，不再在此程序中对信件作时间限定。
//2018.07.26更新：为了适配模糊搜索对所有信件聚类名称的获取，取消聚类表只输出top50的限制。
import java.io.FileInputStream
import java.text.SimpleDateFormat
import java.util.{Date, Properties}
import org.ansj.splitWord.analysis.NlpAnalysis
import org.apache.spark.graphx.Graph
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.{SparseVector => SV}
import org.apache.spark.{SparkConf, SparkContext}
object hotTopic {
  def main(args: Array[String]): Unit = {
    /*读取配置文件信息*/
    val dateFormat1: SimpleDateFormat = new SimpleDateFormat("yyyyMMddHHmmss")
    val dateFormat2: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
    val DATE=dateFormat2.format(new Date()) //获取年月日YYYYMMDD
    val filePath =System.getProperty("user.dir") //设定为jar包的绝对路径 在IDE中运行时为project的绝对路径。System.getProperty("user.dir")为取当前路径。
    val properties = new Properties()
    //val ipstream= new FileInputStream("//opt/govern/idea/00_put/conf/conf.properties")//将配置文件的路径写死，适配现在提交到集群时无法动态获取user.dir的情况
    val ipstream= new FileInputStream(filePath+"//conf.properties")
    properties.load(ipstream)
    var dataSource=properties.getProperty("dataSource")+DATE+"/part-00000"
    println("数据源位置："+dataSource)//读取键为dataSource的key值
    val targetPathForMtable = properties.getProperty("targetPathForMtable","null")    //properties.getProperty("ddd","没有值")//如果ddd不存在,则返回第二个参数
    val targetPathForCtable = properties.getProperty("targetPathForCtable","null")
    val targetPathForMainNode=properties.getProperty("targetPathForMainNode","null")
    val targetPathForNodeConnection=properties.getProperty("targetPathForNodeConnection","null")
    val appName = properties.getProperty("appName")
    val masterValue = properties.getProperty("masterValue")
    val datePeriodOfData = properties.getProperty("datePeriodOfData")
    val nature1=properties.getProperty("nature1")
    val nature2=properties.getProperty("nature2")
    val nature3=properties.getProperty("nature3")
    val nature4=properties.getProperty("nature4")
    //生成转化时间格式的转化器
    /*数据初始化*/
    val sim = 0.5
    val sparkconfig = new SparkConf().setAppName(appName).setMaster(masterValue).set("spark.testing.memory","471859201")
    val ctx = new SparkContext(sparkconfig)
    //加载过滤数据(id,title,content)
    val input = ctx.textFile(dataSource).map(x => x.split("\\|").toSeq).filter(x =>x(1).length > 1 && x(2).length > 1)

    //分词(id,title,[words])
    val splitWord = input.map(x => (x(0), x(1), NlpAnalysis.parse(x(2)).toStringWithOutNature(" ").split(" ").toSeq))
    splitWord.take(5).foreach(x =>println(x))

    //聚类初始化 计算文章向量(id,(id,content,title))
    val init_rdd = input.map(a => {
      (a(0).toLong, a)
    })

    println("init_rdd")
    init_rdd.take(5).foreach(x =>println(x))
    init_rdd.cache()

    //计算TF-IDF特征值
    val hashingTF = new HashingTF(Math.pow(2, 18).toInt)
    //计算TF
    val newSTF = splitWord.map(x => (x._1, hashingTF.transform(x._3))
    )
    newSTF.cache()

    //构建idf model
    val idf = new IDF().fit(newSTF.values)
    //将tf向量转换成tf-idf向量
    val newsIDF = newSTF.mapValues(v => idf.transform(v)).map(a => (a._1, a._2.toSparse))

//    val indice=hashingTF.indexOf("我")
//    print("indice")
//    println(indice)
//    val idfArray=newSTF.mapValues(v => idf.transform(v)).map(a => (a._2.toSparse)(indice))
//    print("idfArray")
//    idfArray.take(1).foreach(x =>println(x))


    newsIDF.take(10).foreach(x => println("newsIDF==:"+x))
    //构建hashmap索引 ,特征
    val indexArray_pairs = newsIDF.map(a => {
      val indices = a._2.indices
      val values = a._2.values
      val result = indices.zip(values).sortBy(-_._2).take(10).map(_._1)
      (a._1, result)
    })
    //(id,[特征ID])
    indexArray_pairs.cache()

    indexArray_pairs.take(10).foreach(x => println("indexArray_pairs==:"+x._1 + " " + x._2.toSeq))
    //倒排序索引 (词ID,[文章ID])
    val index_idf_pairs = indexArray_pairs.flatMap(a => a._2.map(x => (x, a._1))).groupByKey()
    index_idf_pairs.take(10).foreach(x => println("index_idf_pairs==:"+x._1 + " " + x._2.toSeq))


    val b_content = index_idf_pairs.collect.toMap
    //广播全局变量
    val b_index_idf_pairs = ctx.broadcast(b_content)
    //广播TF-IDF特征
    val b_idf_parirs = ctx.broadcast(newsIDF.collect.toMap)


    //相似度计算 indexArray_pairs(id,[特征ID]) b_index_idf_pairs( 124971 CompactBuffer(21520885, 21520803, 21521903, 21521361, 21524603))
    val docSims = indexArray_pairs.flatMap(a => {
      //将包含特征的所有文章ID
      var ids: List[Long] = List()
      //存放文章对应的特征
      var idfs: List[(Long, SV)] = List()

      //遍历特征，通过倒排序索引取包含特征的所有文章,除去自身
      a._2.foreach(b => {
        ids = ids ++ b_index_idf_pairs.value.get(b).get.filter(x => (!x.equals(a._1))).map(x => x.toLong).toList
      })
      //b_idf_parirs(tf-idf特征),遍边文章，获取对应的TF-IDF特征
      ids.foreach(b => {
        idfs = idfs ++ List((b, b_idf_parirs.value.get(b.toString).get))
      })
      //获取当前文章TF-IDF特征
      val sv1 = b_idf_parirs.value.get(a._1).get

      import breeze.linalg._
      //构建当前文章TF-IDF特征向量
      val bsv1 = new SparseVector[Double](sv1.indices, sv1.values, sv1.size)
      //遍历相关文章
      val result = idfs.map {
        case (id2, idf2) =>
          val sv2 = idf2.asInstanceOf[SV]
          //对应相关文章的特征向量
          val bsv2 = new SparseVector[Double](sv2.indices, sv2.values, sv2.size)
          //计算余弦值
          val cosSim = bsv1.dot(bsv2) / (norm(bsv1) * norm(bsv2))
          (a._1, id2, cosSim)
      }
      // 文章1，文章2，相似度
      result.filter(a => a._3 >= sim.toFloat)
    })


    //取出所有，有相似度的文章
    val vertexrdd = docSims.map(a => {
      (a._2.toLong, a._1.toLong)
    })


    //构建图
    val graph = Graph.fromEdgeTuples(vertexrdd, 1)
    val graphots = Graph.graphToGraphOps(graph).connectedComponents().vertices
    //参考图的关系边
    graphots.take(10).foreach(x => println("graphots==:"+x))`


    //关联数据数据详情
    val simrdd = init_rdd.join(graphots).map(a => {
      (a._2._2, (a._2._1, a._1))
    })
    //查看关联数据详情后的数据结构数据内容
    simrdd.take(10).foreach(x =>println("simrdd==:"+x))

    //过滤掉文章篇数小于3的组。类标识分组，并按类内的评论数据降序排列
    val simrddtop = simrdd.groupByKey().sortBy(-_._2.size).collect()
    val simrdd2 = ctx.parallelize(simrddtop, 18)

    println("simrdd2")
    simrdd2.take(5).foreach(x =>println(x))

    //生成聚类后所有主题和主题id构成的rdd
    val clusterTable = simrdd2.map(x=> {
      val titles = x._2.map(x => x._1(1)).toArray  //数据拓展需要修改索引位置，使用“索引”两字搜索需要修改的位置，需要修改x._1(6)中6的值，代表标题名。
      val title = mostSimilartyTitle(titles)
      (x._1,title)
  })

    val clusterTableOut = clusterTable.map(x=>x._1+"|"+x._2+"|")
    clusterTableOut.take(5).foreach(x=>println("话题="+x))

    clusterTableOut.repartition(1).saveAsTextFile(targetPathForCtable+DATE)     //将所有数据集中在一个分区后以文本形式输出

    val acticleMapType = simrdd.map(x=>x._2._2+"|"+x._1+"|")
    acticleMapType.take(5).foreach(x=>println("文章与对应的话题："+x))
    acticleMapType.repartition(1).saveAsTextFile(targetPathForMtable+DATE)     //将所有数据集中在一个分区后以文本形式输出




//构造拓扑图数据
    val splitWordForTopo = simrdd.map(x=>(x._1.toString,x._2._2,NlpAnalysis.parse(x._2._1(2))))
    //val splitWordForTopo = simrdd.map(x=>(x._1.toString,NlpAnalysis.parse(x._2._1(2))))
    //val addIfidf=splitWordForTopo.join(newsIDF).map(x=> (x._1, x._2._1, x._2._2, x._2._2))
    val idfArray=newsIDF.collect().toMap  //转换为映射类型，方面使用键值对取值。
    idfArray.take(5).foreach(x =>println(x))
    val splitWordwWithNature= splitWordForTopo.map(a => {
      //将包含特征的所有文章ID
      var wordsAndNature: List[(String,String,String,String,String)] = List()
      val segTerms = a._3.iterator()
      val strbuf = new StringBuffer

      while (segTerms.hasNext()) {
        val tm = segTerms.next
        val strNs = tm.getNatureStr //获取词性
        val strNm: String = tm.getName
        val indice=hashingTF.indexOf(strNm)
        val wordTfidf=idfArray.apply(a._2.toString).apply(indice)
        wordsAndNature=wordsAndNature++ List((a._1.toString,a._2.toString,strNm,strNs,wordTfidf.toString))
      }
      wordsAndNature
    })
    splitWordwWithNature.take(5).foreach(x =>println(x))
    val toFlat=splitWordwWithNature.flatMap(x=>x)//将所有行的数据融合为一行，这样，原本主题分行的数据就成为一行
    toFlat.take(2).foreach(x => println(x))
    //    val rddColumn = ctx.parallelize(toFlat.collect())
    val rddColumn = ctx.parallelize(toFlat.collect()).map(x=>((x._1,x._3,x._4),(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.apply(0))).map(x=>(x._1._1,x._1._2,x._1._3,x._2._1,x._2._2))
    rddColumn.take(20).foreach(x=>println(x))


//人名nr
    //生成单个词性下，所有主题权重前5的关键词，以纵列显示
    val rddnrTopWithTfidf=rddColumn.filter(x=>x._3==nature1).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {
      val len = x._2.length
      var wordsList: List[(String, String, String)] = List()
      if (len > 0) {
        for (a <- 1 to len) {
          wordsList = wordsList ++ List((x._1, x._2.apply(a - 1)._1, x._2.apply(a - 1)._2))
        }
      }
      wordsList
    })
    print("rddnrTopWithTfidf")
    rddnrTopWithTfidf.take(5).foreach(x=>println(x))
    val rddnrTopWithTfidfToFlat=  (rddnrTopWithTfidf).flatMap(x=>x)
    val rddnrTopWithTfidColumn=ctx.parallelize(rddnrTopWithTfidfToFlat.collect())
    print("rddnrTopWithTfidfToFlat")
    rddnrTopWithTfidColumn.collect().foreach(x=>println(x))



    //生成单个词性下，同一主题的前5个关键词，以主题为组行展示
    val rddnrTop=rddColumn.filter(x=>x._3==nature1).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {//取同主题的同属性前5个词，注意sorBy的升序降序
      val len=x._2.length
      var wordsList: List[String] = List()
      if (len>0){
        for( a <- 1 to len){
         wordsList=wordsList++List(x._2.apply(a-1)._1)


        }
      }
      (x._1,wordsList.toSeq)
    })
    print("rddnr")
    rddnrTop.take(5).foreach(x=>println(x))


    //生成单个词性下，同一主题内所有词关键词和主关键词的关联关系
    val rddnrConnection=rddnrTop.map(x=> {
      val len = x._2.length
      var wordsList: List[(String, (String, String))] = List()
      if (len > 1) {
        for (a <- 2 to len) {
          wordsList = wordsList ++ List((x._1,(x._2.apply(0).toString, x._2.apply(a-1 ).toString)))
        }
      }
      wordsList
    })
    print("rddnrConnection")
    rddnrConnection.take(5).foreach(x=>println(x))
    val rddnrConnectionToFlat=  (rddnrConnection).flatMap(x=>x)
    val rddnrConnectionColumn=ctx.parallelize(rddnrConnectionToFlat.collect())
    print("rddnrConnection")
    rddnrConnection.collect().foreach(x=>println(x))



//机构nt

    //生成单个词性下，所有主题权重前5的关键词，以纵列显示
    val rddntTopWithTfidf=rddColumn.filter(x=>x._3==nature2).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {
      val len = x._2.length
      var wordsList: List[(String, String, String)] = List()
      if (len > 0) {
        for (a <- 1 to len) {
          wordsList = wordsList ++ List((x._1, x._2.apply(a - 1)._1, x._2.apply(a - 1)._2))
        }
      }
      wordsList
    })
    print("rddntTopWithTfidf")
    rddntTopWithTfidf.take(5).foreach(x=>println(x))
    val rddntTopWithTfidfToFlat=  (rddntTopWithTfidf).flatMap(x=>x)
    val rddntTopWithTfidColumn=ctx.parallelize(rddntTopWithTfidfToFlat.collect())
    print("rddntTopWithTfidfToFlat")
    rddntTopWithTfidColumn.collect().foreach(x=>println(x))



    //生成单个词性下，同一主题的前5个关键词，以主题为组行展示
    val rddntTop=rddColumn.filter(x=>x._3==nature2).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {//取同主题的同属性前5个词，注意sorBy的升序降序
      val len=x._2.length
      var wordsList: List[String] = List()
      if (len>0){
      for( a <- 1 to len){
        wordsList=wordsList++List(x._2.apply(a-1)._1)
        }
      }
      (x._1,wordsList.toSeq)
    })
    print("rddnt")
    rddntTop.take(5).foreach(x=>println(x))


    //生成单个词性下，同一主题内所有词关键词和主关键词的关联关系
    val rddntConnection=rddntTop.map(x=> {
      val len = x._2.length
      var wordsList: List[(String, (String, String))] = List()
      if (len > 1) {
        for (a <- 2 to len) {
          wordsList = wordsList ++ List((x._1,(x._2.apply(0).toString, x._2.apply(a-1 ).toString)))
        }
      }
      wordsList
    })
    print("rddnrConnection")
    rddntConnection.take(5).foreach(x=>println(x))
    val rddntConnectionToFlat=  (rddntConnection).flatMap(x=>x)
    val rddntConnectionColumn=ctx.parallelize(rddntConnectionToFlat.collect())
    print("rddnrConnection")
    rddntConnection.collect().foreach(x=>println(x))


// 地名ns
    //生成单个词性下，所有主题权重前5的关键词，以纵列显示
    val rddnsTopWithTfidf=rddColumn.filter(x=>x._3==nature3).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {
      val len = x._2.length
      var wordsList: List[(String, String, String)] = List()
      if (len > 0) {
        for (a <- 1 to len) {
          wordsList = wordsList ++ List((x._1, x._2.apply(a - 1)._1, x._2.apply(a - 1)._2))
        }
      }
      wordsList
    })
    print("rddnsTopWithTfidf")
    rddnsTopWithTfidf.take(5).foreach(x=>println(x))
    val rddnsTopWithTfidfToFlat=  (rddnsTopWithTfidf).flatMap(x=>x)
    val rddnsTopWithTfidColumn=ctx.parallelize(rddnsTopWithTfidfToFlat.collect())
    print("rddnsTopWithTfidfToFlat")
    rddnsTopWithTfidColumn.collect().foreach(x=>println(x))


    //生成单个词性下，同一主题的前5个关键词，以主题为组行展示
    val rddnsTop=rddColumn.filter(x=>x._3==nature3).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {//取同主题的同属性前5个词，注意sorBy的升序降序
    val len=x._2.length
      var wordsList: List[String] = List()
      if (len>0){
        for( a <- 1 to len){
          wordsList=wordsList++List(x._2.apply(a-1)._1)
        }
      }
      (x._1,wordsList.toSeq)
    })
    print("rddns")
    rddnsTop.take(5).foreach(x=>println(x))


    //生成单个词性下，同一主题内所有词关键词和主关键词的关联关系
    val rddnsConnection=rddnsTop.map(x=> {
      val len = x._2.length
      var wordsList: List[(String, (String, String))] = List()
      if (len > 1) {
        for (a <- 2 to len) {
          wordsList = wordsList ++ List((x._1,(x._2.apply(0).toString, x._2.apply(a-1 ).toString)))
        }
      }
      wordsList
    })
    print("rddnsConnection")
    rddnsConnection.take(5).foreach(x=>println(x))
    val rddnsConnectionToFlat=  (rddnsConnection).flatMap(x=>x)
    val rddnsConnectionColumn=ctx.parallelize(rddnsConnectionToFlat.collect())
    print("rddnsConnection")
    rddnsConnection.collect().foreach(x=>println(x))

//时间t
    //生成单个词性下，所有主题权重前5的关键词，以纵列显示
    val rddtTopWithTfidf=rddColumn.filter(x=>x._3==nature4).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {
      val len = x._2.length
      var wordsList: List[(String, String, String)] = List()
      if (len > 0) {
        for (a <- 1 to len) {
          wordsList = wordsList ++ List((x._1, x._2.apply(a - 1)._1, x._2.apply(a - 1)._2))
        }
      }
      wordsList
    })
    print("rddofTop")

    print("rddtTopWithTfidf")
    rddtTopWithTfidf.take(5).foreach(x=>println(x))
    val rddtTopWithTfidfToFlat=  (rddtTopWithTfidf).flatMap(x=>x)
    val rddtTopWithTfidColumn=ctx.parallelize(rddtTopWithTfidfToFlat.collect())
    print("rddtTopWithTfidfToFlat")
    rddtTopWithTfidColumn.collect().foreach(x=>println(x))


    //生成单个词性下，同一主题的前5个关键词，以主题为组行展示
    val rddtTop=rddColumn.filter(x=>x._3==nature4).map(x=>(x._1,(x._2,x._5))).groupByKey().map(x=>(x._1,x._2.toArray.sortBy(tfidf=>tfidf._2).take(5).toSeq)).map(x=> {//取同主题的同属性前5个词，注意sorBy的升序降序
    val len=x._2.length
      var wordsList: List[String] = List()
      if (len>0){
        for( a <- 1 to len){
          wordsList=wordsList++List(x._2.apply(a-1)._1)
        }
      }
      (x._1,wordsList.toSeq)
    })
    print("rddt")
    rddnsTop.take(5).foreach(x=>println(x))


    //生成单个词性下，同一主题内所有词关键词和主关键词的关联关系
    val rddtConnection=rddtTop.map(x=> {
      val len = x._2.length
      var wordsList: List[(String, (String, String))] = List()
      if (len > 1) {
        for (a <- 2 to len) {
          wordsList = wordsList ++ List((x._1,(x._2.apply(0).toString, x._2.apply(a-1 ).toString)))
        }
      }
      wordsList
    })
    print("rddtConnection")
    rddtConnection.take(5).foreach(x=>println(x))
    val rddtConnectionToFlat=  (rddtConnection).flatMap(x=>x)
    val rddtConnectionColumn=ctx.parallelize(rddtConnectionToFlat.collect())
    print("rddtConnection")
    rddtConnection.collect().foreach(x=>println(x))


//    //所有的节点和节点大小值
    //节点是拓扑图的各个实体，节点大小是由权重值决定的。
    val nodeAll=rddnrTopWithTfidColumn.union(rddnsTopWithTfidColumn).union(rddntTopWithTfidColumn).union(rddtTopWithTfidColumn)
    val nodeAllForOutput=nodeAll.map(x=>x._2+"|"+x._3+"|"+x._1+"|")
    nodeAllForOutput.repartition(1).saveAsTextFile(targetPathForMainNode+DATE)     //将所有数据集中在一个分区后以文本形式输出
    print("nodeAll")
    nodeAll.collect().foreach(x=>println(x))

//从节点的连接需要关注主节点的
//   //主节点连接
    val nrtemp=rddnrTop.map(x=>(x._1,x._2.apply(0)))
    val nstemp=rddnsTop.map(x=>(x._1,x._2.apply(0)))
    val nttemp=rddntTop.map(x=>(x._1,x._2.apply(0)))
    val ttemp=rddtTop.map(x=>(x._1,x._2.apply(0)))
    val mainNode=nrtemp.union(nstemp).union(nttemp).union(ttemp).groupByKey().map(x=> {
      val len = x._2.toSeq.length
      var wordsList: List[(String, (String, String))] = List()
      if (len > 1) {
        for (a <- 1 to len-1) {
          wordsList = wordsList ++ List((x._1,(x._2.toSeq.apply(a-1).toString, x._2.toSeq.apply(a ).toString)))
        }
      }
      wordsList
    })
    print("mainNode")
    mainNode.collect().foreach(x=>println(x))
    val mainNodeToFlat=  (mainNode).flatMap(x=>x)
    val mainNodeColumn=ctx.parallelize(mainNodeToFlat.collect())
    print("mainNodeColumn")
    mainNodeColumn.collect().foreach(x=>println(x))

// 同词性连接关系
    val sameNatureNode=rddnrConnectionColumn.union(rddnsConnectionColumn).union(rddntConnectionColumn).union(rddtConnectionColumn)
    print("sameNatureNode")
    sameNatureNode.collect().foreach(x=>println(x))

//nodeconnectionAll
    val nodeConnectionAll=mainNodeColumn.union(sameNatureNode)
    val nodeConnectionAllForOutput=nodeConnectionAll.map(x=>x._2._1+"|"+x._2._2+"|"+""+"|"+x._1+"|")
    nodeConnectionAllForOutput.repartition(1).saveAsTextFile(targetPathForNodeConnection+DATE)     //将所有数据集中在一个分区后以文本形式输出
    //在控制台查看聚类的结果
    simrdd2.take(10).foreach(x => {
      val titles = x._2.map(x => x._1(1)).toArray     //数据拓展需要修改索引位置，使用“索引”两字搜索需要修改的位置
      //选取事件主题名
      val title = mostSimilartyTitle(titles)
      println(x._1+"事件----" + title)
      x._2.foreach(x => println(x._1(0) + " " + x._1(1)))   //数据拓展需要修改索引位置，使用“索引”两字搜索需要修改的位置
      })
    }



  /**
    * 相似度比对 最短编辑距离
    * @param s
    * @param t
    * @return
    */
  def ld(s: String, t: String): Int = {
    var sLen: Int = s.length
    var tLen: Int = t.length
    var cost: Int = 0
    var d = Array.ofDim[Int](sLen + 1, tLen + 1)
    var ch1: Char = 0
    var ch2: Char = 0
    if (sLen == 0)
      tLen
    if (tLen == 0)
      sLen
    for (i <- 0 to sLen) {
      d(i)(0) = i
    }
    for (i <- 0 to tLen) {
      d(0)(i) = i
    }
    for (i <- 1 to sLen) {
      ch1 = s.charAt(i - 1)
      for (j <- 1 to tLen) {
        ch2 = t.charAt(j - 1)
        if (ch1 == ch2) {
          cost = 0
        } else {
          cost = 1
        }
        d(i)(j) = Math.min(Math.min(d(i - 1)(j) + 1, d(i)(j - 1) + 1), d(i - 1)(j - 1) + cost)
      }
    }
    return d(sLen)(tLen)
  }

  /**
    *
    * @param src
    * @param tar
    * @return
    */
  def similarity(src: String, tar: String): Double = {
    val a: Int = ld(src, tar)
    1 - a / (Math.max(src.length, tar.length) * 1.0)
  }

  /**
    * 选出一组字符串 中相似度最高的
    * @param strs
    * @return
    */
  def mostSimilartyTitle(strs: Array[String]): String = {
    var map: Map[String, Double] = Map()
    for (i <- 0 until strs.length) {
      for (j <- i + 1 until strs.length) {
        var similar = similarity(strs(i), strs(j))
        if (map.contains(strs(i)))
          map += (strs(i) -> (map.get(strs(i)).get + similar))
        else
          map += (strs(i) -> similar)
        if (map.contains(strs(j)))
          map += (strs(j) -> (map.get(strs(j)).get + similar))
        else
          map += (strs(j) -> similar)
      }
    } //end of for
    if (map.size > 0)
      map.toSeq.sortWith(_._2 > _._2)(0)._1
    else
      ""
  }
}