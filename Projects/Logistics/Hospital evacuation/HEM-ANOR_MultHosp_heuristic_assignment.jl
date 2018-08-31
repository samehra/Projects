using JuMP, Gurobi
m2 = Model(solver =GurobiSolver(TimeLimit=1000))

recHos = readdlm("C:/Users/saura/Google Drive/VT/Spring 18 ISE 6666 - Logistics/Assignments/Assignment 3/Class Case_ANOR MultiHosp_1.txt")
recHos = Array{Int}(recHos)
nPT = 9

@variable(m2, x[1:nEH,1:nRH,1:nPT]>=0, Int)

@objective(m2, Min, sum(recHos[j,i]*x[i,j,p] for i in 1:nEH for j in 1:nRH for p in 1:nPT))


@constraint(m2, transportAll[i in 1:nEH, p in 1:nPT], sum(x[i,j,p] for j in 1:nRH) == W[i,p])
@constraint(m2, beds[j in 1:nRH, p in 1:nPT], sum(x[i,j,p] for i in 1:nEH) <= recHos[j,p+2])

println("Solving problem with ",nT," time intervals")
status = solve(m2)
println("Status: ", status,"; Ofv = ", getobjectivevalue(m2), "; SolveTime: ",getsolvetime(m2))
