using JuMP, Gurobi
m = Model(solver=GurobiSolver(TimeLimit=1000))
# use vrp_data2

dist = zeros(n, n)
k=1
for i = 1:n
    for j = 1:n
            dist[i,j] = arcs[k,3]
            k = k + 1
    end
end

C=8
tRucks = 2
@variable(m, x[1:n,1:n,1:tRucks], Bin)
@variable(m, y[1:n,1:n,1:tRucks] >= 0)


@objective(m, Min, sum(dist[i,j]x[i,j,k] for i in 1:n, j in 1:n, k in 1:tRucks))

@constraint(m, no_ii[i in 1:n, k in 1:tRucks], x[i,i,k] == 0) # do not use i to i arcs
@constraint(m, flow_x[i in 1:n, k in 1:tRucks], sum(x[i,j,k] for j in 1:n) == sum(x[j,i,k] for j in 1:n)) # flow in = flow out for each truck
@constraint(m, depot[k in 1:tRucks], sum(x[1,j,k] for j in 2:n) <= 1) # at most one route per vehicle
@constraint(m, exit_i[i in 2:n], sum(x[i,j,k] for j in 1:n, k in 1:tRucks) >= 1) # each delivery node must be visited

@constraint(m, capacity[i in 1:n, j in 1:n, k in 1:tRucks], y[i,j,k] <= C*x[i,j,k]) # vehicle capacity constraint
@constraint(m, flow_y1[i in 2:n, k in 1:tRucks], sum(y[j,i,k] for j in 1:n) - sum(y[i,j,k] for j in 1:n) >= 0)
@constraint(m, flow_y2[i in 2:n], sum(y[j,i,k] for j in 1:n for k in 1:tRucks) - sum(y[i,j,k] for j in 1:n for k in 1:tRucks) == dmnd[i])
status = solve(m)
println("Status: ", status,"; Ofv = ", getobjectivevalue(m), "; SolveTime: ",getsolvetime(m))

yy = getvalue(y)
xx =  getvalue(x)  #round.(Int64,)


for k in 1:tRucks
    print("Truck: ", k, " ")
    for i in 1:n
        for j in 1:n
            if xx[i,j,k] > 0
                print("(", i, ", ", j, ") ")
            end
        end
    end
println(".")
end
for k in 1:tRucks
    print("Truck: ", k, " ")
    for i in 1:n
        for j in 1:n
            if yy[i,j,k] > 0
                print("(", i, ", ", j, "):", yy[i,j,k], " ",)
            end
        end
    end
    println(".")
end
