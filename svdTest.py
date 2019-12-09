#pinv SVD tester
import numpy as np 
from GPS_Reciever import GPS_Reciever as GPS 

gps = GPS([0,0,0], 0, 0)
A = np.array([[5.0 , 6.0 , 2.0],
              [6.0 , 1.0 , 4.0],
              [2.0 , 4.0 , 7.0]])

B = np.array([[2 , 0],[0 , -4]])

C = np.array([[2],[3],[4]])

D = np.array([[2,5,6,7],[9,3,5,4]])

psuedoInvTest = gps.pinvSVD(A)
puesdoInv = np.linalg.pinv(A)
print("Truth A")
print(puesdoInv)
print("Test A")
print(psuedoInvTest)
psuedoInvTest = gps.pinvSVD(B)
puesdoInv = np.linalg.pinv(B)
print("Truth B")
print(puesdoInv)
print("Test B")
print(psuedoInvTest)
psuedoInvTest = gps.pinvSVD(C)
puesdoInv = np.linalg.pinv(C)
print("Truth C")
print(puesdoInv)
print("Test C")
print(psuedoInvTest)
psuedoInvTest = gps.pinvSVD(D)
puesdoInv = np.linalg.pinv(D)
print("Truth D")
print(puesdoInv)
print("Test D")
print(psuedoInvTest)
