import numpy as np
import math

n = int(input("n:"))
s=(n,n)
result = 0

b = [1.0 for i in range(n)]
x = np.zeros(n)
a = np.zeros(s)


for i in range(n):
    
    for j in range(n):
        if i==j:
            a[i][j]=4
        elif i==j+1 or j==i+1 :
            a[i][j]=-1
            
            

resultL = np.diag([ 1.0 for i in range(n)])
result = np.diag([ 1.0 for i in range(n)])
#list_M = []
result = a
for i in range(n):

    M = np.diag([ 1.0 for i in range(n)],0)
    
    for j in range(i+1,n):
        if abs(result[j][i])>abs(result[i][i]):
            temp = result[j]
            result[j] = result[i]
            result[i] = temp


    for j in range(i+1,n):
        M[j][i] = -result[j][i]/result[i][i]
    #print("a:",a)
    
    #print("M:",M)
    

    resultL = np.dot(M,resultL)
    result = np.dot(M,result) 
    #print("result:",result)
    #print("reultL : ",resultL)

print(a)
U = result
#print(result)    
L = np.linalg.inv(resultL)
print("L:")
for i in range(n):
    for j in range(n):
        print('%f'%L[i][j], end="\t")
    print("")
print("")

print("U:")
for i in range(n):
    for j in range(n):
        print('%f'%U[i][j], end="\t")
    print("")       

x = np.dot(L,U)
print("A",x)