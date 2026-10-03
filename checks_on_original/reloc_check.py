# Pure-python replica of CityGraph generation (8 nodes, p=0.3, weights 10-30) to measure
# how often a legal "virtual departure" is closer to the destination than the real origin.
import random, heapq, statistics as st
def gen(seed, n=8, p=0.3, w=(10,30)):
    random.seed(seed); adj={i:{} for i in range(n)}
    nodes=list(range(n)); random.shuffle(nodes)
    for i in range(1,n):
        u,v=nodes[i-1],nodes[i]; x=random.randint(*w); adj[u][v]=x; adj[v][u]=x
    for u in range(n):
        for v in range(n):
            if u!=v and random.random()<p and v not in adj[u]:
                x=random.randint(*w); adj[u][v]=x; adj[v][u]=x
    return adj
def dij(adj,s):
    d={s:0}; prev={s:None}; pq=[(0,s)]
    while pq:
        du,u=heapq.heappop(pq)
        if du>d[u]: continue
        for v,w in adj[u].items():
            if du+w<d.get(v,1e18): d[v]=du+w; prev[v]=u; heapq.heappush(pq,(d[v],v))
    return d,prev
def main():
    tot=short=0; ratios=[]; ncand=[]; diam=[]; deg=[]; paths=[]
    for seed in range(1000):
        adj=gen(seed); n=len(adj)
        D={s:dij(adj,s) for s in adj}
        diam.append(max(D[s][0][t] for s in adj for t in adj)); deg.append(st.mean(len(adj[i]) for i in adj))
        for o in range(n):
            for t in range(n):
                if o==t: continue
                d0=D[o][0][t]
                # next hop on shortest path o->t
                x=t
                while D[o][1][x]!=o: x=D[o][1][x]
                hops=0; y=t
                while y!=o: y=D[o][1][y]; hops+=1
                paths.append(hops)
                c=[k for k in adj[o] if k!=t and k!=x]; ncand.append(len(c))
                for k in c:
                    tot+=1; r=D[k][0][t]/d0; ratios.append(r); short+= D[k][0][t]<d0
    print("mean degree %.2f, weighted diameter mean %.1f max %d"%(st.mean(deg),st.mean(diam),max(diam)))
    print("mean hops of OD shortest path %.2f"%st.mean(paths))
    print("legal non-origin candidates per OD: mean %.2f, share with zero candidates %.2f"%(st.mean(ncand),sum(c==0 for c in ncand)/len(ncand)))
    print("share of candidates strictly closer to destination than origin: %.3f"%(short/tot))
    print("median remaining-distance ratio candidate/origin: %.2f"%st.median(ratios))
    print("max order battery need ~ diameter*10+10 = %d vs min vehicle battery 2000"%(max(diam)*10+10))


if __name__ == "__main__":
    main()
