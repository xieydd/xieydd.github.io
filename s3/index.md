# S3



# All you need know about S3

### AP 场景下 S3 的最佳实践
来源[论文](https://www.vldb.org/pvldb/vol16/p2769-durner.pdf)
Key Points:
-  price and durability tradeoff: Cloud object storage provides the best durability
guarantees while being the cheapest storage option
- latency: first byte latency 与 total latency 看取舍，1MB 以下对象，两者基本一致，对象越大，两者差距越大
- throughput: medium in 80 ~ 90 Gbit/s
- 最佳请求大小：OLAP 中 8~16MiB 最佳 `requests=throughput*(baseLatency+size*dataLatency)/size`
- 对于 S3，饱和 100 Gibt/s 实例的最佳请求并发度为 ∼200–250, base latency 30 ms, data latency 20 ms/MiB 

### 如何利用好 S3 Express 
来源[博客](https://optiowl.cloud/blog/ultimate-guide-to-s3-express-one-zone-costs) 

Key Points:
- block size 为 512kb 最经济,超过 50MB 的 512KB 块存储文件没有成本优势，因为延迟达到s3 且价格无优势
- 在缓存角度如果 item size >8kb 完全可以替换 dynamodb 这类 key value NOSql 数据库
- s3 express 存储是 s3 standard 的7倍


