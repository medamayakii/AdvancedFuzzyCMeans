using CSV, DataFrames
using Clustering, StatsBase, Distances

# 欠測値がない場合
# wine.csvは欠測値がないデータ
data = Matrix(CSV.read("wine.csv", header=false, DataFrame))[:, 1:end-1]
# クラスタ数Cを2から6としてFCM法を実行
# 評価指標としてBezdekのpartition coefficient（V_PC）とDaveのmodification of the V_PC(V_MPC)を用いた．
for C in 2:6
    result = fuzzy_cmeans(data', C, 2.0, tol=1e-5, maxiter=1000)
    w = result.weights
    vpc = sum(w .^ 2) / length(data[:, end]) # V_PC = \frac{1}{n}\sum_{i=1}^{c}\sum_{j=1}^{n}u_{ij}^2
    vmpc = 1 - (C / (C - 1)) * (1 - vpc) # 1-\frac{c}{c-1}(1-V_PC)
    C == 2 && println("[欠測値がない場合]\n   |     V_PC          ,   V_MPC")
    println("C=$C| $vpc, $vmpc")
end

# 欠測値がある場合
function update_centers!(centers, data, weights, fuzziness, H)
    nrows, ncols = size(weights)
    T = eltype(centers)
    for j in 1:ncols
        num = zeros(T, size(data, 1))
        den = zero(T)
        for i in 1:nrows
            δm = weights[i, j]^fuzziness
            num += δm * (data[:, i] .* H[:, i])
            den += δm
        end
        centers[:, j] = num / den
    end
end

function update_weights!(weights, data, centers, fuzziness, dist_metric, H)
    pow = 2.0 / (fuzziness - 1)
    nrows, ncols = size(weights)
    dists = pairwise(dist_metric, data .* H, centers, dims=2)
    for i in 1:nrows
        for j in 1:ncols
            den = 0.0
            for k in 1:ncols
                den += (dists[i, j] / dists[i, k])^pow
            end
            weights[i, j] = 1.0 / den
        end
    end
end

# wine2.csvは欠測値を20個含むデータ
data = Matrix(CSV.read("wine2.csv", header=false, DataFrame))[:, 1:end-1]

H = similar(data)
for j in eachindex(data)
    ismissing(data[j]) ? H[j] = 0.0 : H[j] = 1.0
end

#　この値を使うことはないけど適当な数値を入れとかないとエラーになるのでなんか入れておく
replace!(data, missing => 99999.9)

fuzziness = 2.0
maxiter = 1000
tol = 1e-5

for C in 2:6
    δ = Inf
    nrows, ncols = size(data')
    iter = 0

    # 初期化
    weights = rand(Float64, ncols, C)
    weights ./= sum(weights, dims=2)
    centers = zeros(nrows, C)
    prev_centers = identity.(centers)

    while iter < maxiter && δ > tol
        update_centers!(centers, data', weights, fuzziness, H')
        update_weights!(weights, data', centers, fuzziness, Euclidean(), H')
        δ = maximum(colwise(Euclidean(), prev_centers, centers))
        copyto!(prev_centers, centers)
        iter += 1
    end
    # V_PC = \frac{1}{n}\sum_{i=1}^{c}\sum_{j=1}^{n}u_{ij}^2
    vpc = sum(weights .^ 2) / length(data[:, end])

    # 1-\frac{c}{c-1}(1-V_PC)
    vmpc = 1 - (C / (C - 1)) * (1 - vpc)
    C == 2 && println("\n[欠測値を処理した場合]\n   |     V_PC          |   V_MPC")
    println("C=$C| $vpc, $vmpc")
end
