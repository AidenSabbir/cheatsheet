########################################################

# Segment Tree

lst = [1, 3, 5, 7, 9, 11]
ARRSIZE = len(lst)
segsize = 1
while segsize < ARRSIZE:
    segsize *= 2

def SEGOP(a, b): 
    return a + b

SEGZERO = 0
segtree = [SEGZERO] * (2 * segsize)

def segupdate(ind, val):
    ind += segsize
    segtree[ind] = val
    ind //= 2
    while ind > 0:
        segtree[ind] = SEGOP(segtree[2 * ind], segtree[2 * ind + 1])
        ind //= 2

def segquery(ell, are):
    ell += segsize
    are += segsize
    res = SEGZERO
    while ell <= are:
        if ell % 2 == 1:
            res = SEGOP(segtree[ell], res)
            ell += 1
        if are % 2 == 0:
            res = SEGOP(res, segtree[are])
            are -= 1
        ell //= 2
        are //= 2
    return res

for i in range(len(lst)):
    segupdate(i, lst[i])

print("Querying range [1, 3]:", segquery(1, 3))
segupdate(1,4)
print("Querying range [1, 3]:", segquery(1, 3))

########################################################

# Binary Search

def BS(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

arr = [2, 3, 4, 10, 40]
target = 10
print("index:", BS(arr, target))

########################################################

# Union-Find Disjoint Set

NUMELEM = 2 # 0 and 1
par = [0] * NUMELEM
rnk = [0] * NUMELEM
for i in range(NUMELEM):
    par[i] = i
    rnk[i] = 1

def dsfind(u):
    if u == par[u]:
        return u
    rt = dsfind(par[u])
    par[u] = rt
    return rt

def dscheck(a, b):
    return dsfind(a) == dsfind(b)

def dsunion(a, b):
    a = dsfind(a)
    b = dsfind(b)
    if rnk[a] < rnk[b]:
        a, b = b, a
    rnk[a] += rnk[b]
    par[b] = a

dsunion(0, 1) # to union elements 0 and 1
print(dscheck(0, 1)) # to check if elements 0 and 1 are in the same set (true)

########################################################

# Prime Factorization

import math as mt
MAXN = 100001
spf = [0 for i in range(MAXN)]
def sieve():
    spf[1] = 1
    for i in range(2, MAXN):
        spf[i] = i
    for i in range(4, MAXN, 2):
        spf[i] = 2
    for i in range(3, mt.ceil(mt.sqrt(MAXN))):
        if (spf[i] == i):
            for j in range(i * i, MAXN, i):
                if (spf[j] == j):
                    spf[j] = i
 
def getFactorization(x):
    ret = list()
    while (x != 1):
        ret.append(spf[x])
        x = x // spf[x]
    return ret
 
# Driver code
sieve()
x = 12246
print("prime factorization for", x, ": ",end = "")
p = getFactorization(x)
for i in range(len(p)):
    print(p[i], end = " ")

########################################################

# Pascals Triangle

MOD = 998244353
PASCALSIZE = 3005
pascal = [[1], [1]]
for i in range(2, PASCALSIZE):
    pascal.append([1])
    for j in range(1, i // 2):
        pascal[i].append((pascal[i - 1][j - 1] + pascal[i - 1][j]) % MOD)
    if i % 2 == 1:
        pascal[i].append((pascal[i - 1][i // 2 - 1] + pascal[i - 1][i // 2]) % MOD)
    else:
        pascal[i].append((pascal[i - 1][i // 2 - 1] * 2) % MOD)

def nCr(n, r):
    if r > n // 2:
        r = n - r
    return pascal[n][r]

n = 10
r = 3
result = nCr(n, r)
print("Combination", n, "choose", r, "mod", MOD, ":", result)

########################################################

# Binary Exponentiation with Mod

MOD = 998244353

def binexp(a, b, m=MOD):
    if b == 0:
        return 1
    res = binexp(a, b // 2, m)
    res = (res * res) % m
    if b % 2 == 1:
        res = (res * a) % m
    return res
  
def moddiv(a, b, m=MOD):
    return (a * binexp(b, m - 2)) % m

# Example use case
a = 2
b = 10
result_exp = binexp(a, b)
print("Binary Exponentiation of", a, "^", b, "mod", MOD, ":", result_exp)
a = 5
b = 3
result_div = moddiv(a, b)
print("Modulo Division of", a, "/", b, "mod",
MOD, ":", result_div)

########################################################

# DFS

NUMVERT = 5
adj = [[] for _ in range(NUMVERT)] # NUMVERT + 1 if starting vertex is 1
edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
for edge in edges:
    u, v = edge
    adj[u].append(v)
    adj[v].append(u)  # comment this line if the graph is directed
visited = [False] * NUMVERT # NUMVERT + 1 if starting vertex is 1
def dfs(u):
    visited[u] = True
    print(u, end=' ')  # Print the vertex being visited
    for v in adj[u]:
        if not visited[v]:
            dfs(v)
dfs(0)

########################################################

# Detect cycles in undirected graph

NUMVERT = 5
adj = [[] for _ in range(NUMVERT + 1)]  # Initialize adjacency list for 1-based indexing
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
for edge in edges:
    u, v = edge
    adj[u].append(v)
    adj[v].append(u)
cycle = False
visited = [False] * (NUMVERT + 1)  # Initialize visited array for 1-based indexing

def dfspar(u, par):
    global cycle
    visited[u] = True
    for v in adj[u]:
        if cycle:
            return
        if v == par:
            continue
        if visited[v]:
            # CYCLE HAS BEEN FOUND
            cycle = True
            return
        dfspar(v, u)

for i in range(1, NUMVERT + 1): # starting vertex 1
    if not visited[i]:
        dfspar(i, i)

print("Cycle exists" if cycle else "No cycle detected")

########################################################

# Detect cycles in directed graph

NUMVERT = 5
adj = [[] for _ in range(NUMVERT + 1)]  # Initialize adjacency list for 1-based indexing
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
for edge in edges:
    u, v = edge
    adj[u].append(v)
cycle = False
topo = []

# state replaces visited; 0 = not visited, 1 = currently processing, 2 = done
state = [0] * (NUMVERT + 1)  # Initialize state array for 1-based indexing

def dfstopo(u):
    global cycle
    state[u] = 1
    for v in adj[u]:
        if cycle:
            return
        if state[v] == 1:
            # CYCLE HAS BEEN FOUND
            cycle = True
            return
        if state[v] == 0:
            dfstopo(v)
    state[u] = 2
    topo.append(u)  # vertices are added in reversed topological order

for i in range(1, NUMVERT + 1): # starting index 1
    if state[i] == 0:
        dfstopo(i)

if not cycle:
    topo.reverse()
    print("Topological order:", topo)
else:
    print("Cycle detected in the graph")

########################################################

#BFS

from collections import deque

NUMVERT = 5
SOURCE = 0 # 1 if start vertex 1
adj = [[] for _ in range(NUMVERT)]

edges = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)]
for edge in edges:
    u, v = edge
    adj[u].append(v)
    adj[v].append(u)  # comment this line if the graph is directed

q = deque([])
visited = [False] * (NUMVERT) # NUMVERT+1 if start vertex 1
dist = [-1] * (NUMVERT) # NUMVERT+1 if start vertex 1

visited[SOURCE] = True
dist[SOURCE] = 0
q.append(SOURCE)

print(SOURCE,end=' ')
while len(q) > 0:
    u = q.popleft()
    for v in adj[u]:
        if not visited[v]:
            visited[v] = True
            print(v,end=' ')
            dist[v] = dist[u] + 1
            q.append(v)
print()
print("Distances from source vertex", SOURCE, ":")
for i in range(NUMVERT): # 1,NUMVERT+1 if start vertex 1
    print("Vertex", i, "- Distance:", dist[i])

########################################################

# Dijkstra's shortest path

import heapq

NUMVERT = 5
SOURCE = 0 # 1 for 1 based indexing
MAXEDGEWEIGHT = 1000
INF = NUMVERT * MAXEDGEWEIGHT

adj = [[(1, 2), (2, 4)], [(3, 5)], [(4, 3)], [(4, 1)], []] # Edge vertex 0(1 if 1 based indexing) is connected to 1 with weight 2

par = [INF] * (NUMVERT)  # NUMVERT+1 for 1 based indexing
dist = [INF] * (NUMVERT)  # NUMVERT+1 for 1 based indexing
par[SOURCE] = -1
dist[SOURCE] = 0
visited = [False] * (NUMVERT) # NUMVERT+1 for 1 based indexing
pq = [(0, SOURCE)]

while pq:
    d, u = heapq.heappop(pq)
    if visited[u]:
        continue
    visited[u] = True
    for v, w in adj[u]:  # u-1 for 1 based indexing
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            par[v] = u
            heapq.heappush(pq, (dist[v], v))

# Printing shortest distances and parent vertices
print("Shortest distances from source vertex", SOURCE, ":")
for i in range(NUMVERT): # 1,NUMVERT+1 for 1 based indexing
    print("Vertex", i, "- Distance:", dist[i], ", Parent:", par[i])
    
########################################################

# Fast io

import io,os,sys
input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline
sys.stdout.write(str() + "\n")

########################################################

# Prime or not

def isPrime(number):
    if number<2:
        return False
    for i in range(2,int(number**0.5)+1):
        if number%i==0:
            return False
    return True

########################################################

# Divisors

def divisors(n):
    divisors = []
    i = 1
    while i <= n**0.5:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    return divisors

########################################################

# Sieve

def Sieve(num):
    prime = [True for i in range(num+1)]
    p = 2
    while (p * p <= num):
        if (prime[p] == True):
            for i in range(p * p, num+1, p):
                prime[i] = False
        p += 1
    arr=[]  # or set s=set()
    for p in range(2, num+1):
        if prime[p]:
            arr.append(p)  # or s.add(p)
    return arr
    
########################################################

# Balanced Bracket

def balance(string):
    stack = []
    map =   { 
            ')': '(',
            '}': '{',
            ']': '['
            }
    for i in string:
        print(stack)
        if i in map.values():
            stack.append(i)
        elif i in map.keys():
            if not stack:
                return False
            top = stack.pop()
            if map[i] != top:
                return False
    return len(stack) == 0
    
########################################################

# Hashing Demo

T = int (input())
base = 26
MOD = 1000000007

for t in range (1, T + 1):
    st = input ()
    
    cnt = 0
    fhash = 0
    bhash = 0
    pw = 1
    ln = len (st)
    for i in range (ln - 1):
        fch = ord (st[i]) - ord ('a')
        bch = ord (st[ln - i - 1]) - ord ('a')
        fhash = (fhash * base + fch) % MOD
        bhash = (bch * pw + bhash) % MOD
        if fhash == bhash:
            cnt += 1
        pw = (pw * base) % MOD
    print (f"Case {t}: {cnt}")

########################################################

# segmented sieve

prime = []
def simpleSieve(limit): 
	mark = [True for i in range(limit + 1)]
	p = 2
	while (p * p <= limit):
		if (mark[p] == True): 
			for i in range(p * p, limit + 1, p): 
				mark[i] = False
		p += 1
	for p in range(2, limit): 
		if mark[p]:
			prime.append(p)
def segmentedSieve(n):
	limit = int(math.floor(math.sqrt(n)) + 1)
	simpleSieve(limit)
	low = limit
	high = limit * 2
	while low < n:
		if high >= n:
			high = n
		mark = [True for i in range(limit + 1)]
		for i in range(len(prime)):
			loLim = int(math.floor(low / prime[i]) * prime[i])
			if loLim < low:
				loLim += prime[i]
			for j in range(loLim, high, prime[i]):
				mark[j - low] = False
		for i in range(low, high):
			if mark[i - low]:
				prime.append(i)
		low = low + limit
		high = high + limit
	return prime

########################################################

# read untill EOF

while True:
    try:
        s=input()
        #code
    except EOFError:
        break
        
########################################################