using JuMP, Gurobi # CPLEX

#recHos = readdlm("C:/Users/saura/Google Drive/VT/Spring 18 ISE 6666 - Logistics/Assignments/Assignment 3/Class Case_ANOR MultiHosp_1.txt")

m = Model(solver =GurobiSolver(TimeLimit=1000))
#m = Model(solver=CplexSolver(TimeLimit=1000))

println(nPT ,", ",nT,", ",nVT)
evacRisk =Array{Float64}(nRH, nPT, nVT, nT)
bedsOpen =Array{Int}(nRH, nPT) # icu, nicu, picu, other

for  i in 1:nRH, p in 1:nPT, v in 1:nVT, t in 1:nT
    evacRisk[i,p,v,t] = t*threatRisk[p] + travelRisk[v,p]*(recHos[i,1]+2)  # +2 is for loading and unloading time
end

println(nPT ,", ",nT,", ",nVT)

@variable(m, x[1:nRH,1:nPT,1:nVT,1:nT]>=0, Int)
@variable(m, y[1:nRH,1:nVT,1:nT]>= 0, Int)

@objective(m, Min, sum(evacRisk[i,p,v,t]*x[i,p,v,t] for i in 1:nRH for p in 1:nPT for v in 1:nVT for t in 1:nT))

@constraint(m, transportAll[p in 1:nPT], sum(x[i,p,k,t] for i in 1:nRH for k in 1:nVT for t in 1:nT) == W[p])
@constraint(m, beds[i in 1:nRH, p in 1:nPT], sum(x[i,p,v,t] for v in 1:nVT for t in 1:nT) <= recHos[i,p+1])
@constraint(m, vehCap[i in 1:nRH, v in 1:nVT, t in 1:nT], sum(x[i,p,v,t] for p in 1:nPT) <= C[v]*y[i,v,t])
@constraint(m, vehLimit[v in 1:nVT, t in 1:nT], sum(y[i,v,(t-f)] for i in 1:nRH for f in 0:2*(recHos[i,1]+1)-1 if t-f >= 1) <= N[v])
@constraint(m, Limit[t in 1:nT], sum(y[i,v,t] for i in 1:nRH for v in 1:nVT) <= L)

println("Solving problem with ",nT," time intervals")
status = solve(m)
println("Status: ", status,"; Ofv = ", getobjectivevalue(m), "; SolveTime: ",getsolvetime(m))


#=
xx =getvalue(x)
yy =getvalue(y)
duration = 0
for t in 1:nT
    if sum(xx[:,:,:,1:t]) == sum(xx[:,:,:,:]) duration = t end
    if duration == t break end
end
println("Patients Evacuated: ", sum(xx[:,:,:,:]),"; Duration: ", duration, " (",duration*interval/60,")")
println("x1 x2 x3 x4 x5 x6 x7 x8 x9 y1 y2 y3 y4")
for t in 1:nT
    #if sum(xx[:,:,:,t]) >0 println(t," ",sum(xx[:,:,:,t])) end
    println(t," ", sum(xx[:,1,:,t])," ",sum(xx[:,2,:,t])," ",sum(xx[:,3,:,t])," ",sum(xx[:,4,:,t])," ",sum(xx[:,5,:,t])," ",sum(xx[:,6,:,t])," ",sum(xx[:,7,:,t])," ",sum(xx[:,8,:,t])," ",sum(xx[:,9,:,t])," ",sum(yy[:,1,t])," ",sum(yy[:,2,t])," ",sum(yy[:,3,t])," ",sum(yy[:,4,t]))
end
=#
