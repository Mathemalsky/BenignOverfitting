using LinearAlgebra
using Plots

# generate the data for the example
function generateSamples(n, k, mu::Float64)
  # generate theta
  theta = rand(Float64, k)
  @. theta = (theta -0.5) * 10
  
  # generate y with noise
  y = Vector{Float64}(undef, n+k -1)
  for j in 1:n
    y[j] = 1 * (rand(Float64) -0.5)
  end
  for j in n+1:n+k-1
    y[j] = 0
  end
  
  # generate X, update y
  supportPoints = rand(Float64, n)
  @. supportPoints = (supportPoints - 0.5)
  supportPoints = sort(supportPoints)
  X = Matrix{Float64}(undef, k, n+k-1)
  
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

function evalPolynomial(grid, coeff)
  k = length(coeff)
  len = length(grid)
  f = zeros(Float64, len)
  for i in 1:k
    for j in 1:len
      f[j] *= grid[j]
      f[j] += coeff[k + 1 - i]
    end
  end
  return f
end

function plotAll(X::Matrix{Float64}, y::Vector{Float64}, theta::Vector{Float64}, n, k)
  x = Vector{Float64}(undef, n)
  val = Vector{Float64}(undef, n)
  for j in 1:n
    x[j] = X[2,j]
    val[j] = y[j]
  end
  plot(x, val ,label="", seriestype=:scatter) #xlims=[x[1], x[n]]
  
  grid = collect(range(x[1], step=0.01, stop=x[n]))
  f = evalPolynomial(grid, theta)
  plot!(grid, f, label="theta")
  
  theta2 = X' \ y # solve in least squares sense
  f2 = evalPolynomial(grid, theta2)
  plot!(grid, f2, label="regression theta")
end

function wrap(n, k, mu::Float64)
  X, y, theta = generateSamples(n, k, mu)
  plotAll(X, y, theta, n, k)
end
