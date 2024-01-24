---
title: "Efficient Stepping Algorithms and Implementations for Parallel Shortest Paths"
---

SPAA 2021

[Paper](./SPAA_2021_Parallel%20Shortest%20Paths.pdf)

[Slides](./SPAA_2021_Parallel%20Shortest%20Paths_marp.html)

## 总结

单源最短路（SSSP）问题是图论中的经典问题，对其理论与实现的研究都具有重大意义。目前，大部分并行SSSP算法基于Δ-Stepping，即以Δ为步长进行Dijkstra算法，并在子过程中进行Bellman-Ford算法。然而，超参数Δ的选择与图的结构、权重分布及具体实现相关，而且对性能的影响极大。在本文中，作者提出了惰性批量优先队列（LAB-PQ），并基于现有的并行SSSP算法，构建了新的Stepping算法框架，实现了超参数不敏感的并行SSSP算法。新算法在理论上具有较低复杂度，实现后在多种不同的负载下均表现出较高的性能。

## LAB-PQ

- 数据结构：`(id, key)`，id为唯一标识

- `UPDATE(id)`：惰性更新
- `EXTRACT(theta)`：返回key值不大于theta的所有id，该函数是“屏障”，执行调用前所有更新，且不能与其他函数并行。
- 扩展：Reduce

## Stepping算法框架

```python
[Input]: graph G = (V,E,w) and source s
[Output]: distances d(·) from s

d = [+inf] * V.size()
Q = LABPQ(d)
d[s] = 0
Q.update(s)
while Q.size() > 0 :
  for u in Q.extract(extdist()) : #parallel
    for v in G.neighbor(u) : #parallel
      if d[v] < d[u] + w(u,v) :
        d[v] = d[u] + w(u,v)
        Q.update(v)
  finishcheck()
```

| 算法                |   extdist   |         finishcheck          |
| ------------------- | :---------: | :--------------------------: |
| Dijkstra            | $min(Q[v])$ |              -               |
| Bellman-Ford        |  $+\infty$  |              -               |
| Δ-Stepping          |  $i\Delta$  | `if Q.min() >= iΔ : i = i+1` |
| Δ^*^-Stepping (new) |  $i\Delta$  |              -               |
| 𝜌-Stepping (new)    |  Q中前𝜌个   |              -               |

## LAB-PQ实现

- 基于锦标赛树的LAB-PQ实现

  ![锦标赛树](./_SPAA_2021_Parallel%20Shortest%20Paths.assets/锦标赛树.png)

  ```python
  def _mark(id, newflag):
    t = T.leaf(id)
    t.inQ = newflag
    while t != T.root and TestAndSet(t.parent.renew, newflag) :
      t = t.parent
  
  def _sync(t):
    if t.is_leaf() :
      return Q[t.id] if t.inQ else +inf
    if t.renew == 0 :
      return t.k
    t.renew = 0
    leftKey, rightKey = _sync(t.left), _sync(t.right) #parallel
    t.k = min(leftKey, rightKey)
    return t.k
  
  def _extract_from(theta, t):
    if t.is_leaf() :
      if Q[t.id] <= theta :
        mark(t.id, 0)
        return [t.id]
      return []
    if t.k > theta :
      return []
    leftSeq, rightSeq = _extractfrom(theta, t.left), _extract_from(theta, t.right) #parallel
    return leftSeq + rightSeq
  
  def update(id):
    _mark(id, 1)
  
  def extract(theta):
    _sync(T.root)
    return _extract_from(theta, T.root)
  ```

- 在实现中以树状数组形式存储

## 实现细节优化

- 稀疏-密集优化：在邻居较少时，使用数组存储id；在邻居较多时，使用位标识与每个邻居是否有边。
- 无向图的优化：算法框架第10行，优先使用v的邻居进行松弛。
- 𝜌-Stepping中𝜌的选择：部分采样排序，按比例选择阈值
- 邻居较多时的优化：邻居较多时可能导致同步更新负担过大，提前进行同步更新。

## 评价

- 主要贡献：LAB-PQ的提出与无锁的实现。
- 次要贡献：对现有的并行SSSP算法进行了框架梳理，并提出了新的超参数不敏感的SSSP算法。
- 个人思考：先对需要实现的功能进行抽象，再进行数据结构或算法的设计。
