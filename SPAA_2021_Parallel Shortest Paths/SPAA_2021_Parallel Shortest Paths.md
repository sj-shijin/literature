# Efficient Stepping Algorithms and Implementations for Parallel Shortest Paths

###### SPAA 2021





## 总结

单源最短路（SSSP）问题是图论中的经典问题，对其理论与应用的研究都具有重大意义。目前，多数并行SSSP基于Δ-Stepping，即以Δ为步长进行Dijkstra算法，并在子过程中进行Bellman-Ford算法。然而，超参数Δ的选择与图结构，权重分布及算法实现相关，且对算法性能影响极大。作者基于现有并行SSSP算法，提出了一个新的Stepping算法框架，实现了两种新的Stepping算法，在理论与时间中都十分高效。