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

### S3 Express One Zone

### 如何利用好 S3 Express One Zone 
结合[HackNews](https://news.ycombinator.com/item?id=38449827)，[博客](https://optiowl.cloud/blog/ultimate-guide-to-s3-express-one-zone-costs) 以及文章[S3 Express is All You Need](https://www.warpstream.com/blog/s3-express-is-all-you-need)， 我们得出以下结论：
1. S3 Express One Zone 比 S3 Standard 的存储价格高7倍，不适合数据湖场景下的“主存储”。
2. 对于 512KiB 的对象进行 PUT， GET 操作是最省成本的
3. 超过 50MB 的 512KB 块存储文件没有成本优势，因为延迟达到 S3 Standard 且价格无优势
4. One Zone 意味着如果希望数据可以跨 zone 高可用，需要跨 zone 复制，传输有一定的成本
5. Express 也不适合压缩层，因为压缩前对象大概率不会是 512KB 以下
6. 作为 Standard 上的缓存层，当对象小于 645KB 是有性能收益的，但是如果超过那你就需要考虑成本和性能的权衡了
7. 完全可以作为像 dynamodb 这类 key value NOSql 数据库的缓存层
8. 不支持 s3 lifecycle policy
9. 20ms 的延迟，可以作为 replication layer
