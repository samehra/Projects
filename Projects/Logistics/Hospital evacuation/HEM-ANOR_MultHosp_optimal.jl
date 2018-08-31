using JuMP, Gurobi # CPLEX
m = Model(solver =GurobiSolver(TimeLimit=1000))
#m = Model(solver=CplexSolver(TimeLimit=1000))

println(nPT ,", ",nT,", ",nVT)
evacRisk =Array{Float64}(nEH, nRH, nPT, nVT, nT)
bedsOpen =Array{Int}(nRH, nPT) # icu, nicu, picu, other

for  i in 1:nEH, j in 1:nRH, p in 1:nPT, v in 1:nVT, t in 1:nT
    evacRisk[i,j,p,v,t] = t*threatRisk[p] + travelRisk[v,p]*(recHos[j,i]+2)  # +2 is for loading and unloading time
end

println(nPT ,", ",nT,", ",nVT)

@variable(m, x[1:nEH,1:nRH,1:nPT,1:nVT,1:nT]>=0)
@variable(m, y[1:nEH+nRH,1:nEH+nRH,1:nVT,0:nT]>= 0,  Int)
@variable(m, z[1:nEH,1:nVT,1:nT]>= 0)

@objective(m, Min, sum(evacRisk[i,j,p,v,t]*x[i,j,p,v,t] for i in 1:nEH for j in 1:nRH for p in 1:nPT for v in 1:nVT for t in 1:nT))

@constraint(m, transportAll[i in 1:nEH, p in 1:nPT], sum(x[i,j,p,v,t] for j in 1:nRH for v in 1:nVT for t in 1:nT) == W[i,p])
@constraint(m, beds[j in 1:nRH, p in 1:nPT], sum(x[i,j,p,v,t] for i in 1:nEH for v in 1:nVT for t in 1:nT) <= recHos[j,p+2])
@constraint(m, vehCap[i in 1:nEH, j in 1:nRH, v in 1:nVT, t in 1:nT], sum(x[i,j,p,v,t] for p in 1:nPT) <= C[v]*y[i,j,v,t])
@constraint(m, numVeh[v in 1:nVT, t in 1:nT], sum(z[i,v,t] for i in 1:nEH) <= N[v])
@constraint(m, vehFlow1[i in 1:nEH, v in 1:nVT, t in 1:nT], z[i,v,t] + sum(y[j,i,v,(t-f)] for j in 1:nRH for f in 0:(recHos[j,i]) if t-f >= 1) + y[i,i,v,t-1] - y[i,i,v,t] - sum(y[i,j,v,t] for j in 1:nRH) == 0)
@constraint(m, vehFlow2[j in 1:nRH, v in 1:nVT, t in 1:nT], sum(y[i,j,v,(t-f)] for i in 1:nEH for f in 0:(recHos[j,i]+2) if t-f >= 1) + y[j,j,v,t-1] - y[j,j,v,t] - sum(y[j,i,v,t] for i in 1:nEH) == 0)
@constraint(m, Limit[i in 1:nEH, t in 1:nT], sum(y[i,j,v,t] for j in 1:nRH for v in 1:nVT) <= L)
@constraint(m, vehLimit[v in 1:nVT, t in 1:nT], sum(y[i,j,v,(t-f)] for i in 1:nEH for j in 1:nRH for f in 0:2*(recHos[j,i]+1)-1 if t-f >= 1) <= N[v])

println("Solving problem with ",nT," time intervals")
status = solve(m)
println("Status: ", status,"; Ofv = ", getobjectivevalue(m), "; SolveTime: ",getsolvetime(m))


#=
xx =getvalue(x)
yy =getvalue(y)
zz = getvalue(z)
tt = collect(Iterators.flatten(xx))
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
  915
  1377
  1395
  1485
  1557
  1898
  1914
  1986
  2034
  2048
  2076
  3705
  5796
  5828
  5856
  6879
  8019
  9097
  9682
 12557
 79442
 79981
=#
