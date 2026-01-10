# Graph Algorithms Cheat Sheet (Engineering-Grade)

## 0) Graph Theory Introduction

### Core objects
- **Graph** `G = (V, E)`
  - `|V| = n` vertices, `|E| = m` edges
- **Directed** vs **Undirected**
- **Weighted** vs **Unweighted**
- **Simple** graph vs **Multigraph** (parallel edges) vs **Self-loops**
- **Path**: sequence of edges; **simple path**: no repeated vertices
- **Cycle**: path that starts/ends same vertex
- **Connected components** (undirected) / **SCCs** (directed)
- **Degree**
  - Undirected: `deg(v)`
  - Directed: `outdeg(v)`, `indeg(v)`

### Representations (pick for asymptotics + cache locality)
- **Adjacency List**: best default for sparse graphs (`m ~ O(n)`)
  - Iteration over neighbors is `O(deg(v))`
  - Total traversal `O(n + m)`
- **Edge List**: best for algorithms scanning all edges repeatedly (Bellman-Ford, Kruskal)
  - Sequential memory access (good cache locality)
- **Adjacency Matrix**: `O(n^2)` memory; only for dense graphs or when `n` is small and constant-bounded
- **CSR (Compressed Sparse Row)**: adjacency list with contiguous neighbor arrays
  - Best cache locality; avoids pointer chasing in hot paths

### Engineering invariants
- Vertex IDs normalized to `[0, n-1]`
- Input sanity:
  - Bounds-check endpoints
  - Verify no unbounded reads; cap `n, m` per memory budget
  - Use checked arithmetic for `m` sizes, distance sums, `n*n` in matrices
- Determinism:
  - If output depends on traversal order, sort adjacency or preserve input order consistently

---

## 1) Problems in Graph Theory (Pattern → Tool)

| Problem Type | Typical Constraint | Tool |
|---|---:|---|
| Reachability / components | unweighted | DFS/BFS |
| Shortest path (unweighted) | edges weight = 1 | BFS |
| Shortest path (non-negative weights) | `w >= 0` | Dijkstra |
| Shortest path (negative edges) | no negative cycles | Bellman-Ford |
| All-pairs shortest path | `n <= ~500` typical | Floyd-Warshall |
| Topological order | DAG | Kahn / DFS topo |
| Longest path | DAG only (general is NP-hard) | DP on topo |
| Bridges / articulation points | undirected | Tarjan low-link |
| SCCs | directed | Tarjan SCC / Kosaraju |
| Eulerian path/circuit | degree constraints | Hierholzer |
| MST | undirected, connected | Prim / Kruskal |
| Max flow / matching | capacities | Dinic / Edmonds-Karp |
| TSP | `n <= ~20..25` for DP | Bitmask DP |

---

## 2) Depth First Search (DFS)

### Use when
- Reachability, connected components
- Cycle detection (directed/undirected variants)
- Topological sort (via postorder)
- Bridges / articulation points (as a framework)

### Complexity
- Time: `O(n + m)`
- Space: `O(n)` for stack/visited; recursion depth can be `O(n)` (risk stack overflow)

### Edge cases
- Disconnected graphs: must loop over all vertices
- Self-loops / parallel edges (especially for bridge logic)
- Deterministic output: neighbor iteration order matters

### Pseudocode (iterative to avoid recursion limits)
```text
// DFS_ITERATIVE(G, start):
//   // G adjacency list: for each vertex, contiguous neighbor list preferred (CSR)
//   // visited: boolean[n] initialized false
//   stack: stack of (v, next_index)
//   push (start, 0); visited[start] = true
//   while stack not empty:
//     (v, i) = top
//     if i == degree(v):
//       // finished v => postorder hook here if needed
//       pop
//       continue
//     u = adj[v][i]
//     top.next_index++
//     if not visited[u]:
//       visited[u] = true
//       push (u, 0)
//
// Notes:
//   - This form gives both preorder (when first discovered) and postorder (when popped).
//   - For directed graphs, "visited" prevents infinite loops.
//   - For cycle detection in directed graphs, add color/state: 0=unseen,1=active,2=done.
```

---

## 3) Breadth First Search (BFS)

### Use when
- Unweighted shortest paths (each edge cost = 1)
- Level-order traversal, minimum-edge paths
- Multi-source shortest paths (push multiple starts)

### Complexity
- Time: `O(n + m)`
- Space: `O(n)` queue

### Edge cases
- Disconnected: BFS from all sources or loop all nodes
- Large graphs: use ring-buffer queue for cache locality
- Deterministic parents: fixed neighbor order

### Pseudocode
```text
// BFS(G, sources):
//   dist: int[n], init INF
//   parent: int[n], init -1
//   queue: FIFO
//   for s in sources:
//     dist[s] = 0
//     push s
//   while queue not empty:
//     v = pop_front
//     for u in adj[v]:
//       if dist[u] == INF:
//         dist[u] = dist[v] + 1
//         parent[u] = v
//         push u
//
// Notes:
//   - dist is shortest number of edges from nearest source.
//   - parent reconstructs shortest path tree.
//   - Use checked_add for dist[v] + 1 if dist can approach integer max (rare but sanitize).
```

---

## 4) BFS Grid Shortest Path (4/8-neighbor)

### Use when
- Grid maze shortest path, uniform cost
- Obstacles + bounds checks

### Complexity
- Time: `O(R*C)` (each cell processed once)
- Space: `O(R*C)` dist/visited

### Edge cases
- Multiple starts/targets
- Walls, unreachable target
- Non-rectangular input (validate row lengths)
- Coordinate encoding: avoid overflow in `id = r*C + c` with checked multiplication

### Pseudocode (multi-source)
```text
// GRID_BFS(grid[R][C], starts, is_blocked):
//   dist: int[R][C] = INF
//   queue
//   for (sr, sc) in starts:
//     if not is_blocked(sr, sc):
//       dist[sr][sc] = 0
//       push (sr, sc)
//   dirs = [(+1,0),(-1,0),(0,+1),(0,-1)] // or add diagonals
//   while queue not empty:
//     (r, c) = pop_front
//     for (dr, dc) in dirs:
//       nr = r + dr; nc = c + dc
//       if nr,nc out of bounds: continue
//       if is_blocked(nr, nc): continue
//       if dist[nr][nc] == INF:
//         dist[nr][nc] = dist[r][c] + 1
//         push (nr, nc)
//
// Notes:
//   - Use integer bounds checks before indexing.
//   - For path reconstruction, store parent cell.
```

---

## 5) Topological Sort (DAG)

### Use when
- Dependency resolution (prereqs)
- Scheduling tasks with constraints
- Enables DP on DAG (shortest/longest paths)

### Preconditions
- Graph must be a DAG for a full topological ordering
- If cycle exists: detect and return error variant (not partial silently)

### Algorithm A: Kahn’s (in-degree queue)
- Time: `O(n + m)`
- Space: `O(n)`
- Deterministic: use min-heap / ordered queue if you need lexicographically smallest topo order (cost: `O((n+m) log n)`)

```text
// TOPO_KAHN(G):
//   indeg[n]=0
//   for v in 0..n-1:
//     for u in adj[v]: indeg[u]++
//   queue = all v with indeg[v]==0
//   order = empty list
//   while queue not empty:
//     v = pop_front
//     append v to order
//     for u in adj[v]:
//       indeg[u]--
//       if indeg[u]==0: push u
//   if len(order) != n:
//     // cycle exists => return Error(CYCLE_DETECTED)
//   return order
//
// Notes:
//   - indeg decrement must be consistent; parallel edges increment/decrement multiple times (correct).
```

### Algorithm B: DFS postorder
- Also `O(n + m)`
- Needs cycle detection with colors

---

## 6) Shortest / Longest Path on a DAG

### Use when
- Directed acyclic graph only
- Supports negative weights safely (no cycles)

### Complexity
- Time: `O(n + m)` after topo sort
- Space: `O(n)`

### Edge cases
- Unreachable nodes remain INF / -INF
- Longest path: initialize to `-INF`, careful with overflow when adding weights

### Pseudocode
```text
// DAG_SHORTEST_PATH(G, topo, source):
//   dist[n]=INF; dist[source]=0
//   for v in topo:
//     if dist[v]==INF: continue
//     for (u, w) in adj[v]:
//       // relax edge v->u
//       cand = dist[v] + w  // checked_add to prevent overflow
//       if cand < dist[u]: dist[u]=cand
//   return dist
//
// DAG_LONGEST_PATH(G, topo, source):
//   dist[n]=-INF; dist[source]=0
//   for v in topo:
//     if dist[v]==-INF: continue
//     for (u, w) in adj[v]:
//       cand = dist[v] + w  // checked_add
//       if cand > dist[u]: dist[u]=cand
//   return dist
//
// Notes:
//   - For longest path, requires DAG; otherwise NP-hard in general graphs.
```

---

## 7) Dijkstra’s Shortest Path

### Use when
- Weighted graph with **non-negative** edge weights `w >= 0`
- Single-source shortest paths

### Complexity (binary heap)
- Time: `O((n + m) log n)`; practically `O(m log n)`
- Space: `O(n + m)`
- Comparator costs: heap comparisons are frequent; keep heap keys simple (distance, vertex)

### Engineering notes
- Prefer adjacency in contiguous memory (CSR)
- Use 64-bit distances if weights sum can exceed 32-bit
- Avoid decrease-key complexity by pushing duplicates; skip stale heap entries by checking `if d != dist[v] continue`

### Edge cases
- Multiple edges, self-loops: safe
- Zero-weight edges: safe
- Negative edge => invalid; return Error(NEGATIVE_EDGE_FOUND)

### Pseudocode (source code equivalent, but pseudocode)
```text
// DIJKSTRA(G, source):
//   dist[n]=INF; dist[source]=0
//   parent[n]=-1
//   heap = min-heap of (dist, vertex)
//   push (0, source)
//   while heap not empty:
//     (d, v) = pop_min
//     if d != dist[v]:
//       // stale entry due to duplicate pushes
//       continue
//     for (u, w) in adj[v]:
//       if w < 0:
//         // must fail fast; Dijkstra assumes non-negative weights
//         return Error(NEGATIVE_EDGE_FOUND)
//       cand = d + w  // checked_add
//       if cand < dist[u]:
//         dist[u] = cand
//         parent[u] = v
//         push (cand, u)
//   return (dist, parent)
//
// Notes:
//   - For deterministic parent choice in ties, define tie-break: if cand==dist[u], keep smaller parent id.
//   - If you need actual path, reconstruct by following parent[] from target to source.
```

---

## 8) Bellman-Ford

### Use when
- Graph may have negative edges
- Need to detect negative cycles reachable from source

### Complexity
- Time: `O(n*m)` worst-case (only acceptable when bounded; otherwise too slow)
- Space: `O(n)`

### Edge cases
- Negative cycle not reachable from source: shouldn’t invalidate other distances
- For full-cycle marking: after detecting relaxation on nth iteration, propagate “-INF” effect along reachable edges

### Pseudocode
```text
// BELLMAN_FORD(edge_list, n, source):
//   dist[n]=INF; dist[source]=0
//   parent[n]=-1
//   // relax edges up to n-1 times
//   for i in 1..n-1:
//     changed = false
//     for each edge (a, b, w):
//       if dist[a]==INF: continue
//       cand = dist[a] + w  // checked_add
//       if cand < dist[b]:
//         dist[b]=cand
//         parent[b]=a
//         changed = true
//     if not changed: break
//
//   // detect negative cycle reachable from source
//   neg[n]=false
//   for each edge (a, b, w):
//     if dist[a]==INF: continue
//     cand = dist[a] + w
//     if cand < dist[b]:
//       neg[b]=true
//
//   // propagate neg-cycle influence (nodes whose shortest path is undefined => -INF)
//   // do n times to saturate reachability
//   for i in 1..n:
//     for each edge (a, b, w):
//       if neg[a]: neg[b]=true
//
//   return (dist, neg, parent)
//
// Notes:
//   - If neg[v]==true then "shortest" is -infinity (arbitrarily low).
//   - Must not silently output dist[v] in that case; return an explicit variant.
```

---

## 9) Floyd–Warshall (All-Pairs Shortest Paths)

### Use when
- Need all-pairs shortest paths and `n` is small (typical `n <= 400..800` depending on time limits)
- Dense graphs or when adjacency matrix is natural

### Complexity
- Time: `O(n^3)` (forbidden for unbounded datasets; only when `n` is explicitly small-bounded)
- Space: `O(n^2)`

### Edge cases
- Negative edges allowed
- Negative cycle detection: if `dist[i][i] < 0` after algorithm, negative cycle exists (affects paths through that cycle)

### Pseudocode
```text
// FLOYD_WARSHALL(n, dist):
//   // dist is n x n matrix:
//   // dist[i][j]=0 if i==j, weight(i->j) if edge exists, else INF
//   for k in 0..n-1:
//     for i in 0..n-1:
//       if dist[i][k]==INF: continue
//       for j in 0..n-1:
//         if dist[k][j]==INF: continue
//         cand = dist[i][k] + dist[k][j]  // checked_add
//         if cand < dist[i][j]:
//           dist[i][j]=cand
//   // negative cycle check:
//   for i in 0..n-1:
//     if dist[i][i] < 0: return Error(NEGATIVE_CYCLE)
//   return dist
//
// Notes:
//   - Loop order k-i-j is standard; keep dist rows contiguous for cache locality.
//   - Can store next[i][j] to reconstruct paths (update when relax).
```

---

## 10) Bridges and Articulation Points (Undirected)

### Use when
- Identify edges whose removal disconnects graph (**bridges**)
- Identify vertices whose removal disconnects graph (**articulation points**)

### Preconditions
- Undirected graph
- Must handle disconnected graphs: run DFS from all unvisited vertices

### Complexity
- Time: `O(n + m)`
- Space: `O(n)`

### Key idea (Tarjan low-link)
- `tin[v]`: discovery time
- `low[v]`: lowest `tin` reachable from `v` via:
  - zero or more tree edges + at most one back edge

### Edge cases
- Parallel edges: can invalidate naive “parent edge” skip logic
  - Must distinguish edges by unique edge-id, not just parent vertex
- Root articulation rule differs

### Pseudocode
```text
// BRIDGES_ARTICULATION(G):
//   timer=0
//   tin[n]=-1
//   low[n]=0
//   is_art[n]=false
//   bridges = empty list
//
//   DFS(v, parent_edge_id):
//     tin[v]=low[v]=timer; timer++
//     children=0
//     for each (to, edge_id) in adj[v]:
//       if edge_id == parent_edge_id: continue
//       if tin[to] != -1:
//         // back edge
//         low[v] = min(low[v], tin[to])
//       else:
//         children++
//         DFS(to, edge_id)
//         low[v] = min(low[v], low[to])
//         // bridge condition
//         if low[to] > tin[v]:
//           bridges.add(edge_id)
//         // articulation condition (non-root)
//         if parent_edge_id != -1 and low[to] >= tin[v]:
//           is_art[v]=true
//     // root articulation
//     if parent_edge_id == -1 and children >= 2:
//       is_art[v]=true
//
//   for v in 0..n-1:
//     if tin[v]==-1: DFS(v, -1)
//   return (bridges, is_art)
//
// Notes:
//   - Edge-id is mandatory to handle multiedges correctly.
//   - For self-loops: treated as back edge to itself; does not create bridges.
```

---

## 11) Tarjan’s Strongly Connected Components (SCC) (Directed)

### Use when
- Compress directed graph into SCC DAG
- Solve reachability/cycle structure questions
- 2-SAT, dominance of cycles, etc.

### Complexity
- Time: `O(n + m)`
- Space: `O(n)`

### Edge cases
- Disconnected directed graph: run from all nodes
- Deterministic SCC numbering depends on adjacency order

### Pseudocode
```text
// TARJAN_SCC(G):
//   timer=0
//   disc[n]=-1      // discovery index
//   low[n]=0
//   on_stack[n]=false
//   st = stack
//   scc_id[n]=-1
//   scc_count=0
//
//   DFS(v):
//     disc[v]=low[v]=timer; timer++
//     st.push(v); on_stack[v]=true
//     for to in adj[v]:
//       if disc[to]==-1:
//         DFS(to)
//         low[v]=min(low[v], low[to])
//       else if on_stack[to]:
//         low[v]=min(low[v], disc[to])
//     // root of SCC
//     if low[v]==disc[v]:
//       while true:
//         x=st.pop()
//         on_stack[x]=false
//         scc_id[x]=scc_count
//         if x==v: break
//       scc_count++
//
//   for v in 0..n-1:
//     if disc[v]==-1: DFS(v)
//   return (scc_id, scc_count)
//
// Notes:
//   - disc is strictly increasing; low tracks reachable earliest active node.
//   - Output SCC graph can be built by scanning edges and connecting scc_id[u] -> scc_id[v] when different.
```

---

## 12) Travelling Salesman Problem (TSP) via DP (Bitmask)

### Use when
- `n` small (typical `n <= 20..22` practical)
- Need exact optimal Hamiltonian cycle/path

### Complexity
- Time: `O(n^2 * 2^n)`
- Space: `O(n * 2^n)`
- This is exponential; only valid for explicitly small-bounded `n`

### Variants
- **Cycle**: return to start
- **Path**: end anywhere or fixed end

### Edge cases
- Disconnected graph: transitions may be INF
- Overflow: distances can exceed 32-bit

### Pseudocode
```text
// TSP_DP(dist, start):
//   // dist is n x n cost matrix; dist[i][j]=INF if no edge
//   // dp[mask][v] = min cost to start at 'start', visit exactly 'mask', and end at v
//   dp = map or array sized (2^n) x n initialized INF
//   dp[1<<start][start]=0
//   for mask in 0..(2^n - 1):
//     if (mask & (1<<start))==0: continue
//     for v in 0..n-1:
//       if dp[mask][v]==INF: continue
//       if (mask & (1<<v))==0: continue
//       for u in 0..n-1:
//         if (mask & (1<<u))!=0: continue
//         cand = dp[mask][v] + dist[v][u]  // checked_add
//         dp[mask | (1<<u)][u] = min(dp[mask | (1<<u)][u], cand)
//   full = (1<<n)-1
//   ans = INF
//   for v in 0..n-1:
//     cand = dp[full][v] + dist[v][start] // for cycle; omit for path variant
//     ans = min(ans, cand)
//   return ans
//
// Notes:
//   - Store parent pointers if reconstruction required.
//   - Use iterative loops to maximize locality; dp as flat array [mask*n + v].
```

---

## 13) Existence of Eulerian Paths and Circuits

### Eulerian trail facts
- **Undirected**
  - Eulerian **circuit** exists iff:
    - every vertex with nonzero degree is in the same connected component
    - all vertices have even degree
  - Eulerian **path** (not circuit) exists iff:
    - exactly 0 or 2 vertices have odd degree
    - all nonzero-degree vertices connected
- **Directed**
  - Eulerian **circuit** exists iff:
    - for all v: `indeg(v) == outdeg(v)`
    - all vertices with nonzero degree are in one SCC of the underlying graph (usually check connectivity in “edge-present” subgraph)
  - Eulerian **path** exists iff:
    - one start with `outdeg = indeg + 1`
    - one end with `indeg = outdeg + 1`
    - all others `indeg == outdeg`
    - connectivity condition holds

### Edge cases
- Graph with zero edges: trivially Eulerian (path/circuit depends on definition; treat as circuit of length 0)
- Multiple edges and self-loops are allowed; algorithm must track edge usage by edge-id

---

## 14) Eulerian Path Algorithm (Hierholzer)

### Use when
- Need actual Eulerian path/circuit (uses each edge exactly once)

### Complexity
- Time: `O(n + m)`
- Space: `O(n + m)`

### Pseudocode (works with edge-ids; directed/undirected)
```text
// HIERHOLZER(G, start):
//   // G adjacency: list of (to, edge_id)
//   // used[edge_id]=false
//   // it[v] = current index into adjacency list (for O(m) total scanning)
//   it[n]=0
//   st = stack of vertices
//   path = empty list
//   st.push(start)
//   while st not empty:
//     v = st.top()
//     // advance iterator to next unused edge
//     while it[v] < degree(v) and used[ adj[v][it[v]].edge_id ]:
//       it[v]++
//     if it[v] == degree(v):
//       // dead end => add to output
//       path.append(v)
//       st.pop()
//     else:
//       (to, id) = adj[v][it[v]]
//       used[id]=true
//       st.push(to)
//   // path is in reverse
//   reverse(path)
//   // validate: path length should be m+1 (for connected edge-set)
//   if len(path) != m+1: return Error(NOT_EULERIAN_OR_DISCONNECTED)
//   return path
//
// Notes:
//   - For undirected graphs, each undirected edge must appear twice in adjacency but share one edge_id.
//   - Connectivity requirement should be validated beforehand to avoid partial outputs.
```

---

## 15) Prim’s Minimum Spanning Tree (MST)

### Use when
- Undirected weighted graph
- Want MST (or MSF if disconnected)
- Prim is strong for dense graphs with adjacency matrix; also fine with heap for sparse

### Complexity (binary heap)
- Time: `O(m log n)`
- Space: `O(n + m)`

### Edge cases
- Disconnected graph: returns minimum spanning forest (or explicit error if MST requires connected)
- Negative weights: allowed (MST still well-defined)
- Parallel edges: ok, picks cheapest

### Pseudocode (lazy Prim)
```text
// PRIM_LAZY(G):
//   in_mst[n]=false
//   mst_edges = empty
//   total=0
//   for each component root s:
//     if in_mst[s]: continue
//     heap = min-heap of (w, from, to)
//     push all edges from s
//     in_mst[s]=true
//     while heap not empty:
//       (w, a, b) = pop_min
//       if in_mst[b]: continue
//       // add edge to MST
//       in_mst[b]=true
//       mst_edges.add(a,b,w)
//       total += w
//       for (to, w2) in adj[b]:
//         if not in_mst[to]: push(w2, b, to)
//   return (mst_edges, total)
//
// Notes:
//   - Lazy Prim pushes many edges; eager Prim is tighter and often faster.
```

---

## 16) Eager Prim’s MST

### Use when
- Need fewer heap operations than lazy Prim
- Preferable for large sparse graphs

### Complexity
- With binary heap + decrease-key via “push duplicates and skip stale”: `O(m log n)`
- True decrease-key heap: `O(m log n)` but with lower constants

### Pseudocode (eager with best-known edge per vertex)
```text
// PRIM_EAGER(G):
//   key[n]=INF            // best edge weight to connect vertex into MST
//   parent[n]=-1
//   in_mst[n]=false
//   heap = min-heap of (key, v)
//
//   for each component root s:
//     if in_mst[s]: continue
//     key[s]=0
//     push(0, s)
//     while heap not empty:
//       (k, v) = pop_min
//       if in_mst[v]: continue
//       if k != key[v]: continue   // stale
//       in_mst[v]=true
//       for (to, w) in adj[v]:
//         if in_mst[to]: continue
//         if w < key[to]:
//           key[to]=w
//           parent[to]=v
//           push(key[to], to)
//
//   // edges are (parent[v], v) where parent[v]!=-1
//   return (parent, key)
//
// Notes:
//   - key[] + parent[] defines the MST forest.
//   - Determinism in ties: if w==key[to], pick smaller parent id.
```

---

## 17) Max Flow (Ford–Fulkerson Framework)

### Use when
- Compute max `s->t` flow in capacitated directed graph
- Also solves matching and many allocation problems via reductions

### Core residual graph concept
- Each edge has:
  - capacity `cap`
  - flow `f`
  - residual forward capacity: `cap - f`
  - residual backward capacity: `f`
- Augment along an `s->t` path in residual graph

### Pitfalls
- Naive DFS augmenting can be exponential if capacities are irrational or path choices are bad
- For integer capacities, terminates, but can still be too slow
- Prefer Edmonds–Karp, Dinic, or Capacity Scaling in practice

---

## 18) Unweighted Bipartite Matching via Flow

### Use when
- Bipartite graph `L`–`R`, unweighted, maximize matches

### Reduction
- Source `S` to each `u in L` capacity 1
- Each original edge `u->v` capacity 1
- Each `v in R` to sink `T` capacity 1
- Max flow equals max matching

### Complexity choice
- For unweighted bipartite matching, **Hopcroft–Karp** is specialized (`O(m sqrt n)`), but if constrained to “network flow section,” Dinic on unit networks is very fast.

### Edge cases
- Isolated vertices
- Multiple edges (safe; redundant)

---

## 19) “Mice and Owls” via Network Flow (Canonical Reduction)

### Typical structure (general pattern)
- Entities that must be assigned to resources with constraints:
  - mice → holes (capacity per hole)
  - mouse can go to certain holes if reachable within time/distance
- Build bipartite graph (mouse → hole) edges when feasible
- Then run max flow / matching

### Engineering checklist
- Precompute feasibility edges efficiently:
  - If geometric distance: compute squared distances to avoid floating error; compare against squared threshold
  - If grid distance with obstacles: run BFS from holes (multi-source) or from each mouse depending on counts (choose asymptotically)

---

## 20) “Elementary Math” via Network Flow (Canonical Reduction)

### Typical structure (general pattern)
- Assign each pair `(a,b)` one operation result among `{a+b, a-b, a*b}` such that all results are distinct.
- Build bipartite:
  - Left: each pair index `i`
  - Right: each possible result value node
  - Edge if pair i can produce that result via some operation
- Find perfect matching (flow) to assign unique results.

### Edge cases
- Duplicate results across different operations/pairs
- Large result values: compress coordinate values to `[0..K-1]` to keep memory bounded

---

## 21) Edmonds–Karp (Max Flow)

### Use when
- Need a simple, deterministic max-flow
- Graph sizes moderate

### Algorithm
- Ford–Fulkerson where augmenting path is found by BFS in residual graph (shortest in edges)

### Complexity
- Time: `O(n * m^2)` worst-case
- Space: `O(n + m)`
- Too slow for large dense graphs; use Dinic

### Pseudocode
```text
// EDMONDS_KARP(N, edges, s, t):
//   // adjacency stores indices of residual edges; each edge has (to, rev, cap)
//   flow=0
//   while true:
//     parent_v[N]=-1
//     parent_e[N]=-1
//     queue
//     parent_v[s]=s
//     push s
//     while queue not empty and parent_v[t]==-1:
//       v=pop_front
//       for ei in adj[v]:
//         e = edges[ei]
//         if parent_v[e.to]==-1 and e.cap > 0:
//           parent_v[e.to]=v
//           parent_e[e.to]=ei
//           push e.to
//     if parent_v[t]==-1: break // no augmenting path
//     // find bottleneck
//     add=INF
//     v=t
//     while v!=s:
//       ei=parent_e[v]
//       add=min(add, edges[ei].cap)
//       v=parent_v[v]
//     // augment
//     v=t
//     while v!=s:
//       ei=parent_e[v]
//       edges[ei].cap -= add
//       edges[edges[ei].rev].cap += add
//       v=parent_v[v]
//     flow += add
//   return flow
//
// Notes:
//   - Use 64-bit capacities if totals can exceed 32-bit.
//   - Graph must be built with explicit reverse edges for O(1) residual updates.
```

---

## 22) Capacity Scaling (Max Flow)

### Use when
- Large integer capacities; want fewer augmentations than plain FF/EK
- Works well when capacities vary widely

### Idea
- Maintain threshold `Δ` (power of 2)
- Only consider residual edges with capacity ≥ `Δ`
- Find augmenting paths under this restriction; then halve `Δ`

### Complexity (typical statement)
- `O(m^2 log U)` in some analyses for DFS-style; depends on implementation details
- In practice: often speeds up Ford–Fulkerson substantially on large capacities

### Pseudocode (high level)
```text
// CAPACITY_SCALING_MAXFLOW(G, s, t):
//   U = max capacity in network
//   Delta = highest power of 2 <= U
//   flow=0
//   while Delta >= 1:
//     while true:
//       // find any s->t path using only edges with cap >= Delta
//       path = DFS_or_BFS_with_threshold(Delta)
//       if no path: break
//       add = bottleneck_on_path
//       augment(add)
//       flow += add
//     Delta /= 2
//   return flow
//
// Notes:
//   - Determinism: fix adjacency scan order.
//   - Still inferior to Dinic on many benchmarks, but a solid intermediate.
```

---

## 23) Dinic’s Algorithm (Max Flow)

### Use when
- High-performance max flow in general graphs
- Standard competitive + production choice for integral capacities

### Core components
1. **Level graph** via BFS from `s` using residual edges with cap > 0
2. **Blocking flow** via DFS sending flow along level-respecting edges
3. Use `ptr[v]` current-edge pointer to ensure `O(m)` per BFS phase in practice

### Complexity
- General: `O(m n^2)` worst-case (rarely tight in practice)
- Unit networks / bipartite matching: significantly faster (often near `O(m sqrt n)` behavior in practice with structure)
- Space: `O(n + m)`

### Engineering notes
- Use contiguous arrays for edges: `(to, cap, next/rev)` to minimize pointer chasing
- Avoid recursion depth issues in DFS (can be iterative, but recursive often acceptable with bounded stack; sanitize)
- Explicitly return error on overflow in `cap` arithmetic

### Pseudocode
```text
// DINIC(N, s, t):
//   flow=0
//   while BFS_LEVELS():
//     ptr[N]=0
//     while true:
//       pushed = DFS_BLOCKING(s, INF)
//       if pushed==0: break
//       flow += pushed
//   return flow
//
// BFS_LEVELS():
//   level[N]=-1
//   queue
//   level[s]=0; push s
//   while queue not empty:
//     v=pop_front
//     for ei in adj[v]:
//       e=edges[ei]
//       if e.cap>0 and level[e.to]==-1:
//         level[e.to]=level[v]+1
//         push e.to
//   return level[t]!=-1
//
// DFS_BLOCKING(v, pushed):
//   if pushed==0: return 0
//   if v==t: return pushed
//   for i from ptr[v] to degree(v)-1:
//     ptr[v]=i
//     ei=adj[v][i]
//     e=edges[ei]
//     if e.cap>0 and level[e.to]==level[v]+1:
//       tr = DFS_BLOCKING(e.to, min(pushed, e.cap))
//       if tr>0:
//         e.cap -= tr
//         edges[e.rev].cap += tr
//         return tr
//   return 0
//
// Notes:
//   - ptr[v] prevents re-scanning dead edges, critical for performance.
//   - For determinism: adjacency order fixed; stable iteration yields stable flows.
```

---

## 24) Practical Selection Guide (What to Use When)

- **Reachability / components**
  - Use: BFS/DFS
  - If recursion risk: iterative DFS
- **Unweighted shortest path**
  - Use: BFS (multi-source supported)
- **Weighted shortest path**
  - `w >= 0`: Dijkstra
  - Negative edges: Bellman-Ford (and detect cycles)
  - DAG: topo + DP (fastest, supports negative)
- **All-pairs shortest path**
  - Small `n`: Floyd–Warshall
  - Larger sparse: run Dijkstra from each node if `w>=0` (`O(n*m log n)`)
- **DAG ordering / scheduling**
  - Kahn topo (cycle detection built-in)
- **Cycle structure**
  - Directed SCC: Tarjan SCC
  - Undirected bridges/AP: low-link
- **MST**
  - Sparse: Prim eager or Kruskal (not listed, but valid alternative)
  - Dense: Prim with matrix `O(n^2)` only if `n` is small-bounded
- **Max flow / matching**
  - Default: Dinic
  - Educational/simple: Edmonds–Karp (but beware `O(n*m^2)`)

---

## 25) Cross-Cutting Edge Cases & Correctness Invariants

- **Disconnected graphs**
  - Many algorithms require looping all nodes (DFS/BFS/bridges/SCC/topo for full coverage)
- **Parallel edges**
  - Bridges/articulation/Euler need **edge-id** to avoid “parent vertex” ambiguity
- **Self-loops**
  - Shortest paths: harmless
  - Bridges: never a bridge
  - Euler: contributes 2 to degree in undirected? (implementation-defined; treat carefully)
- **Overflow**
  - Distances: use 64-bit; checked add when summing weights
  - Matrices: `n*n` memory sizing must be checked
- **Negative cycles**
  - Only Bellman-Ford / Floyd–Warshall can detect reliably; Dijkstra must reject negative edges
- **Deterministic behavior**
  - Fix adjacency order; define tie-breaks in relaxations; avoid hash-map iteration order for outputs
- **Complexity discipline**
  - Avoid `O(n^2)` unless `n` explicitly bounded (same for `O(n^3)` Floyd, `O(2^n)` TSP)
  - Prefer contiguous memory (CSR / flat edge arrays) for cache locality

---