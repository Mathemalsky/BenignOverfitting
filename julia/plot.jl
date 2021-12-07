using Plots

# generate the data for the example
function generateSamples(n, k, mu::Float32)
  # generate theta
  theta = rand(Float32, k)
  @. theta = (theta -0.5) * 10
  
  # generate y with noise
  y = Vector{Float32}(undef, n+k -1)
  for j in 1:n
    y[j] = 0.5 * (rand(Float32) -0.5)
  end
  for j in n+1:n+k-1
    y[j] = 0
  end
  
  # generate X, update y
  supportPoints = rand(Float32, n)
  @. supportPoints = (supportPoints - 0.5)
  supportPoints = sort(supportPoints)
  X = Matrix{Float32}(undef, k, n+k-1)
  
  for j in 1:n
    X[1,j] = 1
    y[j] = theta[1]
    for i in 2:k
      X[i,j] = X[i-1, j] * supportPoints[j]
      y[j] += X[i,j] * theta[i]
    end
  end
  
  # identidy matrix multiplied with vector of weights, first row 
  for j in n+1:n+k-1
    X[1,j] = 0
    for i in 1:k
      if (j - n + 1 == i) 
        X[i,j] = sqrt(mu * (j-n))
      end
    end
  end
  
  return X, y, theta
end

function plotAll(X::Matrix{Float32}, y::Vector{Float32}, theta::Vector{Float32}, n, k)
  x = Vector{Float32}(undef, n)
  val = Vector{Float32}(undef, n)
  for j in 1:n
    x[j] = X[2,j]
    val[j] = y[j]
  end
  plot(x, val ,label="", seriestype=:scatter, xlims=[x[1], x[n]])
  
  grid = collect(range(x[1], step=0.01, stop=x[n]));
  len = length(grid)
  f = zeros(Float32, len)
  for i in 1:k
    for j in 1:len
      f[j] *= grid[j]
      f[j] += theta[k + 1 - i]
    end
  end
  
  plot!(grid, f, label="")
end


function wrap(n, k, mu::Float32)
  X, y, theta = generateSamples(n, k, mu)
  plotAll(X, y, theta, n, k)
end
