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

@variable(m, 0 <= x[1:n,1:n] <= 2, Int)
@variable(m, y[1:n,1:n] >= 0)
@variable(m, 0 <=tRucks <= 2, Int)

@objective(m, Min, sum(dist[i,j]x[i,j] for i in 1:n for j in 1:n))
#@objective(m, Min, tRucks)

@constraint(m, no_ii[i=1:n], x[i,i] == 0)

@constraint(m, sum(x[1,j] for j in 1:n) == tRucks)
@constraint(m, exit_i[i in 2:n], sum(x[i,j] for j in 1:n) >= 1) # each delivery node must be visited
@constraint(m, flow_x[i in 1:n], sum(x[j,i] for j in 1:n) == sum(x[i,j] for j in 1:n))
@constraint(m, capacity[i in 1:n, j in 1:n], y[i,j] <= C*x[i,j]) # route must exit each node once
@constraint(m, flow_y[i in 2:n], sum(y[j,i] for j in 1:n) - sum(y[i,k] for k in 1:n) == dmnd[i])

status = solve(m)
println("Status: ", status,"; Ofv = ", getobjectivevalue(m), "; SolveTime: ",getsolvetime(m))

yy = getvalue(y)
xx =round.(Int64, getvalue(x))
tRucks1 = convert(Int16, getvalue(tRucks))
println("Trucks = ",tRucks1)

for i in 1:n
    for j in 1:n
        if xx[i,j] > 0
            print("(", i, ", ", j, "):", xx[i,j], " ",)
        end
    end
end
println(".")
for i in 1:n
    for j in 1:n
        if yy[i,j] > 0
            print("(", i, ", ", j, "):", yy[i,j], " ",)
        end
    end
end
