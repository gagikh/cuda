# Day 13: Cache Behavior and Optimization

## Objectives
- Deepen understanding of shared-memory bank conflicts and how to remove them
- Understand L1/L2 cache behavior and persistence hints
- Use `__ldg` to hint read-only global memory access
- Promise non-aliasing with `const __restrict__`, and know why that is what lets the compiler emit read-only-cache loads on its own
- Understand LRU cache eviction and how to bias it with per-instruction cache operators
- Amortize per-thread fixed costs with thread coarsening, and see why it trades against occupancy
- Express a kernel's speed as a percentage of theoretical peak bandwidth, and use it to decide when to stop
- Apply the week's techniques to optimize a real kernel

## Key Concepts
- Bank conflicts
- Using L2 cache
- Persistent cache for compiled programs and configuration
- `__ldg` forces the compiler to consider memory read-only
- LRU cache eviction, and per-instruction cache hints (`__ldcs`, `__stcs`, `__ldlu`, ...)
- Thread coarsening
- Memory-bound vs. compute-bound; achieved bandwidth as a fraction of peak

## Definitions · Սահմանումներ

*Terms introduced today. Same text as the matching entries in [GLOSSARY.md](../GLOSSARY.md).*

**Cache line** — L1-ում տեղ հատկացնելու 128-բայթանոց միավորը։

**Sector** — 32-բայթանոց միավոր, որով հիշողությունից տվյալներն իրականում կարդացվում և տեղափոխվում են։ Warp-ի դիմումը հաշվվում է sector-ներով, և չորս sector-ը կազմում են մեկ cache line։ Coalescing-ը մեկ request-ի sector-ների քանակի նվազեցումն է։

**L1** — Ամեն SM-ի cache-ը։ Ֆիզիկապես այն նույն SRAM-ն է, ինչ shared memory-ն։

**L2** — Ամբողջ չիպի cache-ը, որը գտնվում է device-ի հիշողության առջև և ընդհանուր է բոլոր SM-ների համար։ Global memory-ի atomic-ները կատարվում են այստեղ։

**Cache operator** — Հրամանի մակարդակի ցուցում այն մասին, թե load-ը կամ store-ն ինչպես օգտագործի cache-երը։ `__ldg`՝ load միայն կարդալու համար նախատեսված cache-ի ճանապարհով, `__ldcs`՝ streaming load, որի տողը cache-ից դուրս է մղվում առաջինը, `__stcs`՝ streaming store, `__ldlu`՝ վերջին օգտագործման load, որից հետո տողը հեռացվում է cache-ից։

**`__restrict__`** — Ցուցիչի որակիչ, որը կոմպիլյատորին խոստանում է, որ այդ ցուցիչով հասցեագրվող հիշողությունը նույն շրջանակում այլ ցուցիչով չի հասցեագրվում։ `const`-ի հետ միասին (`const T *__restrict__`) այն թույլ է տալիս կոմպիլյատորին արժեքները պահել ռեգիստրներում գրառումների միջև և ինքնուրույն թողարկել read-only cache-ի (`LDG`) load-եր՝ առանց ձեռքով գրված `__ldg()`-ի։ Սա խոստում է, ոչ թե ստուգում. եթե ցուցիչներն իրականում համընկնում են, վարքն undefined է։

**LRU** — Least recently used. Դուրս մղման կանոն, որով cache-ից հեռացվում է ամենավաղուց չօգտագործված տողը։ Իրական cache-երն այս կանոնին հետևում են մոտավորապես։

**Thread coarsening** — Ամեն thread մշակում է մեկի փոխարեն մի քանի ելքային տարր։ Արդյունքում մեկ thread-ի հաստատուն ծախսերը (ինդեքսի հաշվարկ, սահմանների ստուգում, shared memory-ի tile-ի բեռնում) կատարվում են մեկ անգամ և բաշխվում են մի քանի տարրի վրա։ Grid-stride loop-ը coarsening-ը գրելու այն ձևն է, որը չի խախտում coalescing-ը։ Coarsening-ը նվազեցնում է occupancy-ն, ուստի արդյունքը պետք է չափել։

**Swizzling** — Shared memory-ի ինդեքսի վերադասավորում, օրինակ՝ `tile[row][col ^ row]`, որպեսզի միևնույն տրամաբանական սյունն ամեն տողում ընկնի այլ ֆիզիկական bank-ում։ Այն հեռացնում է bank conflict-ները՝ առանց լրացուցիչ սյան։

**Տեսական peak bandwidth** — Bus width-ի, հիշողության clock-ի և մեկ clock-ում փոխանցումների քանակի արտադրյալը՝ հաշված Օր 1-ում ստացած թվերով։ Սա այն վերին սահմանն է, որի հետ համեմատվում է kernel-ը։

**Հասած bandwidth** — Kernel-ի իրականում տեղափոխած բայթերը՝ բաժանած նրա կատարման ժամանակին։ Սովորաբար արտահայտվում է տեսական peak-ի տոկոսով։

**Arithmetic intensity** — Կատարված FLOP-երի քանակը հիշողության տրաֆիկի մեկ բայթի հաշվով։ Այն որոշում է, թե roofline-ի որ մասում է kernel-ը։ Ցածր intensity-ն նշանակում է memory-bound (kernel-ների մեծ մասն այդպիսին է), բարձրը՝ compute-bound։ Tiling-ը բարձրացնում է intensity-ն՝ առանց թվաբանական գործողությունները փոխելու։

**Compute-bound և memory-bound** — Ցույց է տալիս, թե ինչն է սահմանափակում kernel-ի արագությունը՝ հրամանների throughput-ը, թե հիշողության bandwidth-ը։ Դրանից է կախված, թե որ օպտիմալացումները կարող են օգնել։

**Roofline** — Հասանելի առավելագույն արագագործության գրաֆիկը՝ կախված arithmetic intensity-ից։ Այն բաղկացած է bandwidth-ի թեք սահմանից և compute-ի հորիզոնական սահմանից։ Kernel-ի դիրքը գրաֆիկի վրա ցույց է տալիս, թե որ օպտիմալացումը կարող է օգնել՝ հիշողության, թե հրամանների։

## Functions · Ֆունկցիաներ

*Interfaces introduced today. Full consolidated list in [API.md](../API.md); concepts in [GLOSSARY.md](../GLOSSARY.md).*

```c
// Կարդում է միայն կարդալու համար նախատեսված cache-ի ճանապարհով
unsigned char __ldg(const unsigned char *ptr);
float         __ldg(const float *ptr);

// Streaming load՝ տողը cache-ից դուրս է մղվում առաջինը
unsigned char __ldcs(const unsigned char *ptr);
float         __ldcs(const float *ptr);

// Վերջին օգտագործման load՝ դրանից հետո տողը հեռացվում է cache-ից
unsigned char __ldlu(const unsigned char *ptr);
float         __ldlu(const float *ptr);

// Streaming store
void __stcs(unsigned char *ptr, unsigned char value);
void __stcs(float *ptr, float value);

// Նույն ֆունկցիաները կան նաև int-ի, double-ի և վեկտորային տիպերի համար

// L2 persistence՝ stream-ի ատրիբուտով
cudaError_t cudaStreamSetAttribute(cudaStream_t hStream, cudaStreamAttrID attr,
                                   const cudaStreamAttrValue *value);
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);   // cudaLimitPersistingL2CacheSize
cudaError_t cudaCtxResetPersistingL2Cache(void);

// Պատուհանի դաշտերը. base_ptr, num_bytes, hitRatio, hitProp, missProp
// Սահմանները՝ prop.persistingL2CacheMaxSize և prop.accessPolicyMaxWindowSize

// Օր 2-ի ֆունկցիան՝ template-ի TODO 3-ի համար
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                          int blockSize, size_t dynamicSMemSize);
```

## Visual
![Memory hierarchy pyramid: registers (fastest, smallest) at top, then shared memory / L1, then L2, then global memory / VRAM (slowest, largest) at bottom](cache_hierarchy.svg)

Every optimization this day is about the same idea: keep frequently-read data as high in this pyramid as possible, for as long as possible. `__ldg()`, shared-memory swizzling, and L2 persistence hints are three different tools for the same goal.

![XOR swizzling: an 8x8 shared-memory tile where accessing logical column 3 without swizzling always hits bank 3 for every row, but indexing with tile[row][col ^ row] spreads that same logical column across a different bank on every row, with no padding column needed](swizzling.svg)

Padding (Day 5) fixes bank conflicts by wasting a column so the row stride is no longer a multiple of the bank count. Swizzling fixes the same problem without wasting any memory: index shared memory as `tile[row][col ^ row]` instead of `tile[row][col]`. Because XOR is its own inverse, writing and reading with the same swizzle formula is still correct — you just physically scatter each logical column across every bank instead of pinning it to one.

For the hardware behind this pyramid — why shared memory and L1 are the same physical SRAM, why L2 is shared across every SM instead of being per-SM like L1, and where atomic operations actually get resolved — see [ARCHITECTURE.md](../ARCHITECTURE.md).

## Animated
![Three requests traveling from the SM down to L1, L2, and global memory at different speeds, glowing whichever level they land in](memory_traffic.svg)

Most traffic resolves fast, close to the SM (green, ~1s round trip); some reaches L2 (blue, ~3s); a rare few pay the full global-memory round trip (red, ~6s). Same pyramid as `cache_hierarchy.svg`, now showing where requests actually land.

![Side-by-side comparison: coalesced access completing in one transaction versus strided access needing 32 sequential transactions for the same 128 bytes](coalescing_strided.svg)

For a fully interactive, playable version (fire requests on demand, step through) see [`memory_animations.html`](memory_animations.html) — open it locally in a browser, since GitHub's file viewer only shows HTML as source rather than running it.

## `const __restrict__` · Խոստումը, որը բացում է read-only ճանապարհը

`__restrict__`-ը ցուցիչի որակիչ է (C99-ի `restrict`-ը, CUDA-ում գրվում է `__restrict__`)։ Այն կոմպիլյատորին տրվող խոստում է. **այս ցուցիչով հասցեագրվող հիշողությունը նույն շրջանակում ոչ մի այլ ցուցիչով չի հասցեագրվում**։

Տեսնենք, թե ինչու է դա կարևոր.

```c++
__global__ void blur3(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1 || i >= n - 1) return;
    out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3.0f;   // երեք load, մեկ store
}
```

Կոմպիլյատորը չգիտի, որ `out`-ն ու `in`-ը տարբեր բուֆերներ են — դրանք կարող էին լինել նույն զանգվածի մասեր։ Ուստի ամեն գրառումից հետո նա պարտավոր է ենթադրել, որ `in`-ի բովանդակությունը կարող է փոխված լինել։ Հետևանքները.

- ռեգիստրում արդեն կարդացած արժեքը չի կարող վերաօգտագործվել գրառումից հետո՝ պետք է նորից կարդալ
- load-երը չեն կարող խմբավորվել կամ դուրս բերվել ցիկլից
- read-only cache-ի ճանապարհը (`LDG`) չի կարող օգտագործվել

Ավելացնենք երկու բառն էլ.

```c++
__global__ void blur3(const float *__restrict__ in, float *__restrict__ out, int n)
```

Այժմ կոմպիլյատորը գիտի երկու բան՝ `in`-ը kernel-ի ողջ կյանքի ընթացքում չի փոխվում (`const`), և ոչ ոք այլ ցուցիչով դրա մեջ չի գրում (`__restrict__`)։ Այդ պահից նա **ինքն է** թողարկում `LDG` հրաման, այսինքն՝ `__ldg()`-ն ձեռքով գրելու կարիք չկա։ Հենց դրա համար է, որ [`template.cu`](template.cu)-ի `tiled_filter_ldg`-ի պարամետրերը նշված են `const unsigned char *__restrict__`-ով։

**Երկու կեսն էլ պարտադիր են։** Միայն `const`-ը բավարար չէ. այն ասում է, որ *դու* չես գրելու այդ ցուցիչով, բայց ոչ թե որ ուրիշը չի գրելու մեկ այլ ցուցիչով։ Միայն `__restrict__`-ը նույնպես բավարար չէ. կոմպիլյատորը չի իմանա, որ տվյալները kernel-ի համար միայն կարդալու են։

**Սա խոստում է, ոչ թե ստուգում։** Եթե ցուցիչներն իրականում համընկնում են, վարքն undefined է, և ոչ մի տեղ սխալ չի ազդարարվի՝ պարզապես կստանաք սխալ արդյունք։ Այստեղ ստուգողը դուք եք, ոչ թե կոմպիլյատորը։

**Գործնական կանոն.** kernel-ի բոլոր ցուցիչ-պարամետրերը նշեք `__restrict__`-ով, իսկ մուտքայինները՝ նաև `const`-ով։ Ազդեցությունն ամենամեծն է coarsening-ի հետ (ստորև). երբ մեկ thread հաշվում է մի քանի ելքային տարր, ամեն գրառումից հետո մուտքերը նորից կարդալն ուղղակի ավելորդ երթևեկ է ստեղծում, և հենց այն երթևեկը, որ coarsening-ը պետք է խնայեր։

## LRU and Per-Instruction Cache Hints
L1 and L2 are both finite, so when full, something gets evicted to make room for a new line. The hardware's replacement policy is an approximation of **LRU (Least Recently Used)**: evict whichever line hasn't been touched in the longest time. `cudaAccessPolicyWindow` (next section) biases that policy for a whole buffer at once; there's also a finer-grained tool — per-instruction cache operators — that biases it one load or store at a time:

```c++
float x = __ldcs(&input[i]);   // "streaming" read: touched once, evict-first hint
__stcs(&output[i], result);    // "streaming" write: won't be re-read, don't linger in cache
```

![Չորս տողանոց cache՝ MRU-ից LRU դասավորությամբ A, B, C, D։ X-ը կարդալիս երեք դեպքում էլ D-ն դուրս է մղվում, բայց լռելյայն load-ի դեպքում X-ը մտնում է MRU դիրքում, __ldcs-ի դեպքում՝ LRU դիրքում՝ դառնալով հաջորդ զոհը, իսկ __ldlu-ի դեպքում տողն ընթերցումից հետո անվավեր է դառնում և տեղն ազատ է մնում](lru_eviction.svg)

Ուշադրություն երեք տողերի միջև եղած տարբերությանը. **զոհը բոլոր դեպքերում նույնն է** — տեղ բացելու համար դուրս է մղվում LRU դիրքի տողը, `D`-ն։ Cache operator-ը չի փոխում, թե ով է դուրս գալիս. այն փոխում է, թե **նոր տողը որտեղ է մտնում**, և հենց դրանով է որոշվում, թե ինչ կլինի հաջորդ բաց թողնումի ժամանակ։ `__ldcs`-ով բերված `X`-ն ինքն է դառնում հաջորդ զոհը, ուստի `A`, `B`, `C`-ն, որոնք իրոք վերաօգտագործվում են, մնում են cache-ում։

`__ldcs`/`__stcs` are the two you'll reach for most: mark data you're touching exactly once so it doesn't crowd out data your kernel (or a neighboring warp) genuinely reuses. This is a *different* knob from `__ldg()` — `__ldg` changes *which cache* a read uses (the read-only/texture path, Day 11); `__ldcs`/`__stcs` stay on the normal L1/L2 path and change *how eagerly the replacement policy discards the line*. Full intrinsic list (`__ldca`, `__ldcg`, `__ldcs`, `__ldlu`, `__ldcv`, `__stwb`, `__stcg`, `__stcs`, `__stwt`) and PTX mnemonics in [ARCHITECTURE.md](../ARCHITECTURE.md#cache-eviction-hints-lru-and-loadstore-cache-operators).

### L2 persistence՝ `cudaAccessPolicyWindow`

Operator-ները մեկ հրամանի մակարդակի հուշումներ են։ Երբ ուզում ենք **ամբողջ բուֆեր** պահել L2-ում, օգտագործվում է access policy window-ը՝ stream-ի ատրիբուտ, որը գործում է այդ stream-ում գործարկվող բոլոր kernel-ների վրա։

```c++
__global__ void my_kernel(const float *__restrict__ hot,          // կարդացվում է բազմիցս
                          float *__restrict__ streaming, int n)   // անցնում է մեկ անգամ
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        streaming[idx] += hot[idx % HOT_ELEMS];
    }
}
```

```c++
cudaDeviceProp prop{};
CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

// 1. Առանձնացնել L2-ի persisting մասը։ Առանց սրա window-ը պահուստ չունի։
const size_t set_aside = std::min(hot_bytes, (size_t)prop.persistingL2CacheMaxSize);
CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside));

cudaStream_t stream;
CUDA_CHECK(cudaStreamCreate(&stream));

// 2. Նկարագրել պատուհանը
cudaStreamAttrValue attr{};
attr.accessPolicyWindow.base_ptr  = reinterpret_cast<void *>(d_hot);
attr.accessPolicyWindow.num_bytes = std::min(hot_bytes, (size_t)prop.accessPolicyMaxWindowSize);
attr.accessPolicyWindow.hitRatio  = 1.0f;                           // պատուհանի ո՞ր մասն է persisting
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;   // ներսում՝ պահել
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;    // դրսում՝ դուրս մղել առաջինը

// 3. Կիրառել stream-ի վրա
CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));

for (int i = 0; i < iters; ++i)
    my_kernel<<<blocks, threads, 0, stream>>>(d_hot, d_stream, n);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaStreamSynchronize(stream));

// 4. Մաքրել. նախ անջատել պատուհանը, ապա ազատել L2-ի persisting տողերը
attr.accessPolicyWindow.num_bytes = 0;
CUDA_CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
CUDA_CHECK(cudaCtxResetPersistingL2Cache());
```

Չորս բան, որոնք հեշտ է բաց թողնել.

- **`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, …)`-ը պարտադիր է։** Window-ը ցույց է տալիս, թե *որ* հասցեներն են persisting, իսկ այս limit-ը որոշում է, թե L2-ի *որքան* մասն է ընդհանրապես հատկացված persisting տողերին։ Առանց դրա հուշումը գրեթե ոչինչ չի տալիս։
- **Երկու վերին սահման կա, և դրանք տարբեր են։** `prop.persistingL2CacheMaxSize`-ը առանձնացվող մասի առավելագույնն է, `prop.accessPolicyMaxWindowSize`-ը՝ հենց պատուհանի։ Երկուսն էլ պետք է հարցնել, ոչ թե ենթադրել։
- **`hitRatio < 1.0f`-ը պատահականորեն է ընտրում։** Երբ պատուհանն ավելի մեծ է, քան առանձնացված L2-ը, `hitRatio = 0.5f` նշանակում է՝ տողերի մոտավորապես կեսը persisting։ Պատուհանը մեծացնելով և `hitRatio`-ն 1.0 թողնելով՝ ստացվում է thrashing. ավելի լավ է պատուհանը փոքրացնել։
- **Մաքրումը մոռանալը ազդում է հաջորդ kernel-ների վրա։** Persisting տողերը մնում են L2-ում նաև այն բանից հետո, երբ ձեր kernel-ն ավարտվել է, և նեղացնում են մնացած ամեն ինչի համար հասանելի cache-ը։ Այդ պատճառով վերջում `num_bytes = 0` և `cudaCtxResetPersistingL2Cache()`։

**Ուշադրություն վերևի օրինակի չափսին։** `HOT_ELEMS = 1024` նշանակում է 4 ԿԲ, որն առանց որևէ հուշման տեղավորվում է L1-ում — այդպիսի փորձից տարբերություն չեք տեսնի։ Որպեսզի հուշումն իմաստ ունենա, hot տիրույթը պետք է լինի **L1-ից զգալիորեն մեծ**, բայց **առանձնացված L2-ից ոչ մեծ**։ Հենց այս պատուհանը գտնելն է հաջորդ առաջադրանքի բովանդակությունը։

### Առաջադրանք, որտեղ LRU-ն իրոք երևում է

Վերևի նկարը բացատրում է, թե ինչ է անում operator-ը, բայց պնդումը պետք է չափել։ Խնդիրն այնպես է կառուցված, որ cache-ի վարքը դառնա միակ տարբերությունը երկու տարբերակի միջև։

**Խնդիրը՝ batched dot product ֆիքսված հենքային վեկտորի հետ** (իրական գործողություն՝ matched filtering)։ Ունենք `ref[M]` հենքային վեկտորը և `data[B*M]` մեծ զանգվածը։ Ամեն block հաշվում է իր կտորի սկալյար արտադրյալը **նույն** `ref`-ի հետ.

```c++
// block b → out[b]
out[b] = Σ(j = 0 … M-1)  data[b*M + j] * ref[j];
```

Երկու հոսքը սկզբունքորեն տարբեր են.

- **`data`** — `B*M` տարր, ամեն մեկը կարդացվում է **ճիշտ մեկ անգամ**։ Cache-ում պահելն անիմաստ է։
- **`ref`** — ընդամենը `M` տարր, բայց կարդացվում է **բոլոր `B` block-երի կողմից**։ Եթե այն մնում է L2-ում, DRAM-ից բերվում է մեկ անգամ՝ `M` տարր։ Եթե streaming `data`-ն այն դուրս է մղում, DRAM-ից բերվում է `M·B` տարր։

Հենց սա է LRU-ի խնդիրը. լռելյայն load-ի դեպքում ամեն նոր `data` տող մտնում է MRU դիրքում և քայլ առ քայլ դուրս է հրում `ref`-ը, թեև հենց `ref`-ն է վերաօգտագործվողը։

**Քայլերը.**

1. **Չափել L2-ի չափը** — `cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev)`։ Բոլոր հետագա չափսերը արտահայտել դրա մասերով, ոչ թե բացարձակ MB-երով, որպեսզի արդյունքը չկախվի կոնկրետ GPU-ից։
2. **Տարբերակ A՝ լռելյայն** — և՛ `data`-ն, և՛ `ref`-ը սովորական load-երով։
3. **Տարբերակ B՝ operator-ներով** — `data`-ն `__ldcs`-ով (մեկ անգամ է կարդացվում՝ թող առաջինը դուրս մղվի), `ref`-ը `const __restrict__`-ով, որ կոմպիլյատորն ինքը թողարկի read-only load։
4. **Տարբերակ C՝ `cudaAccessPolicyWindow`** — `ref`-ը բացահայտ նշել որպես persisting, `data`-ն թողնել սովորական։ Սա նույն գաղափարն է, բայց ամբողջ բուֆերի մակարդակում։
5. **Սահել `M`-ով** — `M·sizeof(float)` = L2-ի **0.25×, 0.5×, 1×, 2×, 4×**։ `B`-ն ընտրել այնպես, որ `B*M`-ը բոլոր կետերում նույնը մնա, այլապես կչափեք ուղղակի տվյալների ծավալը։
6. **Հաշվել տեսական երթևեկը** — լավագույն դեպքում DRAM-ից բերվում է `B*M + M` տարր, վատագույնում՝ `B*M + B*M`։ Չափված ժամանակից ստացված hասած bandwidth-ը համեմատել այս երկու թվի հետ և ասել, թե որին է մոտ ամեն տարբերակը։

**Ինչ պետք է ստացվի և ինչու։** `M ≈ 0.25×L2`-ի դեպքում երեք տարբերակն էլ գրեթե նույնն են՝ `ref`-ն այնքան փոքր է, որ ողջ է մնում առանց որևէ հուշման։ `M ≈ 1×L2`-ի շրջակայքում բացը ամենամեծն է՝ հենց այստեղ է որոշիչ, թե նոր տողը MRU դիրք է մտնում, թե LRU։ `M ≈ 4×L2`-ի դեպքում բացը նորից փակվում է՝ `ref`-ը ոչ մի կերպ չի տեղավորվում, և հուշումը ոչինչ չի փրկում։

Այս երեք կետից ամենակարևորը վերջին երկուսն են. **cache-ի հուշումը ոչ թե ունիվերսալ արագացում է, այլ գործիք, որը աշխատում է միայն որոշակի տիրույթում**։ Եթե ձեր կորագիծը հարթ է բոլոր կետերում, դա նույնպես արդյունք է. նշանակում է ձեր GPU-ի վրա այդ տիրույթն այլ տեղ է, և պետք է գտնել այն։

## Thread Coarsening

Every kernel so far launches one thread per output element. That's the natural default, and it's often too much parallelism.

Each thread re-pays a fixed cost: index arithmetic, bounds checks, the shared-memory tile loads it shares with its neighbours, and its share of block launch overhead. When that fixed cost is a meaningful fraction of the useful work, giving each thread *several* elements amortizes it. That's **thread coarsening**.

The naive way to write it is wrong in an instructive way:

```c++
// BROKEN: thread 0 takes 0-3, thread 1 takes 4-7 -- lanes are now stride-4,
// so you've traded a small overhead win for uncoalesced access (Day 2). Net loss.
int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
for (int k = 0; k < 4; ++k) if (i + k < n) out[i + k] = f(in[i + k]);
```

The correct form keeps adjacent lanes on adjacent addresses at every step — which is exactly the grid-stride loop from Day 2:

```c++
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int k = i; k < n; k += stride) out[k] = f(in[k]);   // still coalesced
```

So **you already wrote a coarsened kernel on Day 2** without it being named. The knob is the grid size: launch fewer blocks and each thread naturally handles more elements. Try `vector_add_grid_stride` with 64, 256 and 1024 blocks and watch where it peaks.

The bigger win shows up in tiled kernels like this day's `tiled_filter`, where a coarsened thread produces several outputs from *one* set of shared-memory loads:

```
1 output per thread : load tile -> compute 1 result
4 outputs per thread: load tile -> compute 4 results
```

The loads are unchanged, the useful work quadruples. In roofline terms (see below) you've raised arithmetic intensity 4× without touching the algorithm.

**The trade-off is against occupancy.** Each partial result lives in a register, so a coarsened thread needs more of them, and fewer blocks fit per SM. Those two pull in opposite directions, which means this is a knob you *measure* rather than reason about. The usual sweet spot is 2–8 elements per thread; beyond that registers spill to local memory (which is global memory in disguise — see [ARCHITECTURE.md](../ARCHITECTURE.md)) and the gain evaporates. `nvcc -Xptxas -v` reports spills, so it will tell you when you've gone too far.

## Are You Even Memory-Bound? (%-of-peak)

This day times three kernel variants against each other, which answers "which is faster" but not the more useful question: **how much room is left?**

Day 1's `report_device_capabilities()` prints theoretical peak bandwidth. Compute what your kernel actually achieved and divide:

```c++
#include "../common/timer.h"

kernel_timer_t t;
for (int i = 0; i < 100; ++i) { t.start(); tiled_filter_baseline<<<grid, block>>>(...); t.stop(); }
t.report_bandwidth("baseline", 2.0 * width * height);   // one byte in, one byte out
```
```
baseline                    0.087 ms     602.8 GB/s  (84% of 720 GB/s peak)
```

[`common/timer.h`](../common/timer.h) does the arithmetic (and the averaging — a single launch is mostly noise). For a compute-bound kernel like matmul, use `report_tflops()` instead: FLOPs ÷ time, where an M×M×M matmul is `2.0 * M * M * M` operations — the 2 being the multiply *and* the add, which is the convention cuBLAS is benchmarked with.

That one number changes what you do next:

| % of peak | Verdict |
|---|---|
| < 30% | Something structural — check coalescing (Day 2) and occupancy before touching cache hints |
| 50–70% | Normal; the techniques on this page are worth trying |
| > 80% | **Stop.** You're at the hardware limit. `__ldg`, swizzling and cache hints cannot beat the memory bus — the only remaining win is moving fewer bytes |

Add it to the three `printf`s in [`template.cu`](template.cu) and the day stops being "swizzled was 4% faster" and becomes "baseline was already at 84% of peak, so there was never 4× on the table." That's the honest and much more useful conclusion, and it's the discipline behind tasks 94 and 100 in [TASKS.md](../TASKS.md). Full treatment, including arithmetic intensity and where each optimization applies, in [PERFORMANCE.md](../PERFORMANCE.md).

## Resources
https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/

https://cuda-programming.blogspot.com/2013/02/bank-conflicts-in-shared-memory-in-cuda.html

Shared memory swizzling reference:
https://leimao.github.io/images/blog/2024-05-14-CUDA-Shared-Memory-Swizzling/swizzling.png

## Hands-On Task
Optimize transform (the Day 6 image transform kernel), on a real image loaded via `cv::imread` / `cv::cuda::GpuMat`.

## Self-Learning
1. Add `__ldg()` to a read-heavy kernel from an earlier day (e.g. the Day 5 tiled filter) and measure the effect.
2. Fill in `tiled_filter_swizzled` in [`template.cu`](template.cu): use `tile[row][col ^ row]` (both `[TILE_DIM][TILE_DIM]`, no padding column) for every shared-memory read and write, and compare its timing against the padded Day 5 version.
3. Experiment with L2 persistence hints (`cudaAccessPolicyWindow`) on a buffer that's read repeatedly across kernel launches.
4. Optimize the Day 6 image transform kernel using everything from this week (shared memory, texture, `__ldg`, bank-conflict-free layout) and document before/after timings.
5. In one of your filter kernels, write the final output with `__stcs` instead of a plain store (it's written once and never read back inside the kernel), and read the tile's halo region with the default `__ldca` (it's reused by neighboring threads). Measure whether marking the write-once output as streaming changes anything.
6. Add the achieved-bandwidth and %-of-peak calculation to all three `printf`s in [`template.cu`](template.cu). Which variant is closest to peak — and given that number, was there ever as much headroom as the relative timings suggested?
7. Coarsen `tiled_filter_baseline` so each thread computes 2, then 4, then 8 output pixels from the same tile load. Plot time against elements-per-thread and find the peak. Then run `nvcc -Xptxas -v` at each setting and check whether the drop-off coincides with register spilling.
8. Take the Day 2 grid-stride vector add and sweep the grid size (64, 256, 1024, 4096 blocks) with `n` fixed. Explain the shape of the curve in terms of coarsening at one end and occupancy at the other.
9. Կառուցել վերևի batched dot product-ը և երեք տարբերակն էլ չափել՝ լռելյայն, `__ldcs` + `const __restrict__`, և `cudaAccessPolicyWindow`։ `M`-ը սահեցնել L2-ի 0.25×-ից մինչև 4×, `B*M`-ը պահել հաստատուն։ Կառուցել ժամանակի կորագիծը և ցույց տալ, որ հուշման շահույթը առավելագույնն է `M ≈ L2`-ի շրջակայքում և զրոյանում է երկու ծայրում։

## Self-Check
No answers given — these are for you to reason through, or discuss with a classmate/instructor.

1. Why does `__ldg` only help for data the kernel treats as read-only?
2. Why does `col ^ row` swizzling need the row width to be a power of two to cleanly avoid bank conflicts?
3. What's the tradeoff L2 persistence hints (`cudaAccessPolicyWindow`) are making — and when could they make performance *worse* instead of better?
4. Why is the hardware's LRU replacement policy an *approximation* rather than exact LRU, and why does that make cache hints like `__ldcs` useful even when the hardware is "supposed" to figure out reuse patterns on its own?
5. A kernel runs at 88% of theoretical peak bandwidth. Your manager asks for a 2× speedup. What are you allowed to change, and what is definitively off the table?
6. Coarsening raises the work per thread but lowers occupancy. Why isn't there a single right answer, and what property of the kernel decides which way it goes?
7. Why does the "broken" coarsening variant above produce completely correct output while being slower than the uncoarsened kernel it was meant to improve?

## Code Template
See [`template.cu`](template.cu) for a skeleton to start from.

No CUDA GPU on your machine? Run this lab in the [course Colab notebook](https://colab.research.google.com/drive/1zDtYkz8WwD7sOIucSyUoxm2n7RYWJxVZ?usp=sharing) instead — free T4, compute capability 7.5, which is exactly the `-arch=sm_75` the template compiles for. One caveat for this day: Colab's preinstalled OpenCV is CPU-only, so `cv::cuda::GpuMat` will not link. Either build OpenCV with `-DWITH_CUDA=ON` in the notebook, or swap the image I/O for a `cudaMalloc` buffer and a synthetic image — the CUDA content of this day is unchanged either way. Full setup in the [root README](../README.md#-no-cuda-gpu-start-here).
