# Benckmark between Arcface and Centerface

## Environment
- caffe 1.0: disable two `CHECK` in syncedmem
- cuda 8.0 cudnn 6.5
- GTX 1080Ti

## Arcface r34

```shell
feature shape: 1 x 512
feat11 left 5 :-1.34857 -0.00331188 -1.11453 0.319134 -0.162575
similarity:
11 vs 12 : 0.826142  YES
21 vs 22 : 0.569778  NO
11 vs 21 : 0.623811  NO
11 vs 22 : 0.50298   NO
12 vs 21 : 0.66034   NO
12 vs 22 : 0.461413  NO

[Test] feature detecting begin ---------------
[3072x1728] : 232.0000 ms
[3072x1728] : 52.0000 ms
[3072x1728] : 52.0000 ms
face num: 2
[Test] feature extracting begin ---------------
1 : 138.0000 ms
1 : 16.0000 ms
2 : 14.0000 ms
3 : 18.0000 ms
4 : 18.0000 ms
5 : 21.0000 ms
6 : 24.0000 ms
7 : 28.0000 ms
8 : 31.0000 ms
9 : 37.0000 ms
10 : 35.0000 ms
11 : 163.0000 ms
12 : 169.0000 ms
13 : 172.0000 ms
14 : 187.0000 ms
15 : 198.0000 ms
16 : 191.0000 ms
17 : 201.0000 ms
18 : 215.0000 ms
19 : 226.0000 ms
20 : 234.0000 ms
features shape: 512 x 2
[Test] feature similar begin ---------------
 Similar : 0.0000 ms
similarity: 0.6051

# caffe time
I0412 21:11:23.015543 18910 caffe.cpp:408] Average Forward pass: 35.9621 ms.
I0412 21:11:23.015550 18910 caffe.cpp:410] Average Backward pass: 49.2667 ms.
I0412 21:11:23.015558 18910 caffe.cpp:412] Average Forward-Backward: 85.5165 ms.
```

## center face r34

```shell
feature shape: 1 x 512
feat11 left 5 :-10.3503 3.04195 11.0218 0.00557731 0.544645
similarity:
11 vs 12 : 0.846843  YES
21 vs 22 : 0.630423  NO
11 vs 21 : 0.742772  NO
11 vs 22 : 0.520968  NO
12 vs 21 : 0.771157  NO
12 vs 22 : 0.532613  NO

[Test] feature detecting begin ---------------
[3072x1728] : 304.0000 ms
[3072x1728] : 75.0000 ms
[3072x1728] : 62.0000 ms
face num: 2
[Test] feature extracting begin ---------------
1 : 27.0000 ms
1 : 5.0000 ms
2 : 11.0000 ms
3 : 16.0000 ms
4 : 17.0000 ms
5 : 18.0000 ms
6 : 19.0000 ms
7 : 24.0000 ms
8 : 23.0000 ms
9 : 25.0000 ms
10 : 26.0000 ms
11 : 27.0000 ms
12 : 35.0000 ms
13 : 32.0000 ms
14 : 33.0000 ms
15 : 33.0000 ms
16 : 38.0000 ms
17 : 44.0000 ms
18 : 41.0000 ms
19 : 44.0000 ms
20 : 44.0000 ms
features shape: 512 x 2
[Test] feature similar begin ---------------
 Similar : 0.0000 ms
similarity: 0.7231

# caffe time
I0412 21:12:48.238140 19455 caffe.cpp:408] Average Forward pass: 5.5126 ms.
I0412 21:12:48.238147 19455 caffe.cpp:410] Average Backward pass: 9.3732 ms.
I0412 21:12:48.238154 19455 caffe.cpp:412] Average Forward-Backward: 15.0355 ms.
```

## 结论

可以看出 Arcface 相对于 Centerface 得到的相似度更有区分性，但是 Arcface 比 Centerface 在速度上要慢一些。