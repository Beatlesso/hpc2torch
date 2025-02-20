不要直接写下面的代码，会出现启动的 block 数量不足的情况。
```
gather_kernel<T, Tind><<<(grid_dim_x, grid_dim_y), block_size>>>
```
应该改成
```
dim3 grid_dim(grid_dim_x, grid_dim_y);
gather_kernel<T, Tind><<<grid_dim, block_size>>>
```