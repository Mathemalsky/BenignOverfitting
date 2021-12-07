using Plots

# generate the data for the example
function generateSamples(n, k, mu)
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
  suportPoints = rand(Float32, n)
  @. suportPoints = (suportPoints - 0.5) * 10
  X = Matrix{Float32}(undef, k, n+k-1)
  
  @. X[1,:] = 1
  for j in 1:n
    for i in 2:k
      X[i,j] = X[i-1, j] * suportPoints[j]
      y[j] += X[i,j] * theta[i]
    end
  end
  
  # identidy matrix multiplied with vector of weights, first row 
  for j in n+1:n+k-1
    for i in 1:k
      if (i == j-n) 
        x(i,j) = sqrt(mu * (j-n))
      else
        x(i,j) = 0
      end
    end
  end
  
  return X, y, theta
  
end

function plotAll(X, y, theta)
  plot!(X[2,:] ,y',label="", seriestype=:scatter)
end

