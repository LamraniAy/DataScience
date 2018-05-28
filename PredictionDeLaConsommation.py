import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import math
import pandas as pd



Beta1 = -2
Beta0 = 1

muX = 10
sigmaX = 2

muE = 0
sigmaE = 1

n = 100

x = np.random.normal(muX, sigmaX, n)
e = np.random.normal(muE, sigmaE, n)
y = Beta0 + Beta1*x + e




p0=plt.scatter(x, y, c='red')
p0=plt.xlabel('x')
p0=plt.ylabel('y')
p0=plt.title('Nuage des points (x,y)')
plt.show()






def moyenne(valeurs):
    moyenne = 0
    for i in valeurs:
         moyenne = moyenne + i
    moyenne = moyenne * 1/len(valeurs)
    return moyenne
    

def variance(valeurs):
    variance = 0
    somme = 0
    for i in valeurs:
        somme = somme + (i*i)
    variance = ((1/len(valeurs)) * somme) - moyenne(valeurs)*moyenne(valeurs)
    return variance
    
def covariance(valeursX, valeursY):
    covariance = 0
    for i in range(0,len(valeursX)):
        covariance = covariance + (valeursX[i] - moyenne(valeursX)) * (valeursY[i] - moyenne(valeursY))
    covariance = covariance/(len(valeursX)-1)
    return covariance




def coeffcorr(valeursX, valeursY):
    coeffcorr = 0
    coeffcorr = covariance(valeursX, valeursY)/ math.sqrt(variance(valeursX)*variance(valeursY))
    return coeffcorr

moyenneX = moyenne(x)
print("Moyenne de X :"  + str(moyenneX))
moyenneY = moyenne(y)
print("Moyenne de Y :"  + str(moyenneY))

varianceX = variance(x)
print("variance de X :"  + str(varianceX))
varianceY = variance(y)
print("variance de Y :"  + str(varianceY))

covarianceXY = covariance(x,y)
print("covariance de XY :"  + str(covariance(x,y)))

coeffcorrXY = coeffcorr(x,y)
print("coefficient de corrélation de XY :"  + str(coeffcorrXY))



def calculBetaC1(valeursX, valeursY):
    BetaC1 = covariance(valeursX, valeursY) / variance(valeursX)
    return BetaC1

print("Calcul de B1 :"  + str(calculBetaC1(x,y)))


def calculBetaC0(valeursX, valeursY):
    BetaC0 = moyenne(valeursY) - ((covariance(valeursX, valeursY) / variance(valeursX)) * moyenne(valeursX))
    return BetaC0

print("Calcul de B0 :"  + str(calculBetaC0(x,y)))




def droiteReg(valeursX, valeursY):
    valeursYC = calculBetaC0(valeursX, valeursY) + (calculBetaC1(valeursX, valeursY) * x)
    return valeursYC



Xp1 = 10
Yp1 = calculBetaC0(x,y) + calculBetaC1(x,y)*Xp1




Xp2 = 50
Yp2 = calculBetaC0(x,y) + calculBetaC1(x,y)*Xp2


lr = linear_model.LinearRegression()    
lr.fit(x.reshape(100,1),y.reshape(100,1)) 
YPredicted = lr.predict(x.reshape(100,1))


p2=plt.plot(x, droiteReg(x,y)) 
p0=plt.scatter(x, y, c='red')
plt.xlabel('x')
plt.ylabel('y')

plt.title('Manuellement')
plt.show()

p1=plt.plot(x, YPredicted, c='black')
p0=plt.scatter(x, y, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Avec Sklearn')
plt.show()

Beta1Sklearn = lr.coef_[0]
Beta0Sklearn = lr.intercept_

print('Avec sklearn : y = {0} * x + {1} || en noir'.format(Beta1Sklearn, Beta0Sklearn))

print('Manuellement : y = {0} * x + {1} || en bleu'.format(calculBetaC1(x,y), calculBetaC0(x,y)))




xbis = x
ybis = y

xbis[0] = -100
ybis[0] = 0

p3=plt.plot(xbis,droiteReg(xbis,ybis),c='red')
p0=plt.scatter(x, y, c='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Avec valeurs aberrantes")
plt.show()



def SCresidus(Y,Ypred):
    SCresidus = 0
    for i in range(len(Y)):
        SCresidus = SCresidus + ( (Y[i] - Ypred[i]) * (Y[i] - Ypred[i]) )
    return SCresidus


x = np.random.normal(muX, sigmaX, n)


def eAvecVarianceBruit():
    
    ListeSCresidus= []
    VarianceE = []
    
    #Variation de la variance de e de 0 à 99
    for i in range(100):  
        
        e = np.random.normal(muE, i, n)
        y = Beta0 + Beta1*x + e
        yc = droiteReg(x,y)
        
        ListeSCresidus.append(SCresidus(y, yc))
        VarianceE.append(i)
        
    return ListeSCresidus, VarianceE


VariationVarianceBruit = eAvecVarianceBruit()

p4=plt.plot(VariationVarianceBruit[1], VariationVarianceBruit[0], c='grey') 
plt.xlabel('Variance de e')
plt.ylabel('SC Residus')
plt.show()




#Nombre d'observations nn
nn = 200

#On définit les Betaas
Betaa0 = 1
Betaa1 = 1.5
Betaa2 = 2


Betaa = np.array([Betaa0, Betaa1, Betaa2]).reshape(3,1)

#On définit ee
ee = np.random.normal(0, 1, nn).reshape(nn, 1)

#Génération des valeurs des variables explicatives X1, X2
def creationX():
    X = np.ones((nn,1))

    for i in range(2):   # X1, X2
        Z = np.random.normal(15, i + 3, nn) 
        X = np.concatenate((X,Z.reshape(nn,1)),1)
    return X 

X = creationX()

#On définit les valeurs de Y
Y = np.dot(X, Betaa) + ee


#Estimation des coefficients (forme matricielle)
def estimation():
    BetaaE = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)) , np.dot(np.transpose(X),Y))
    YC = np.dot(X , BetaaE)
    return BetaaE , YC.reshape(nn,1)


#Regression linéaire "manuelle"
BetaaE, YC = estimation()


#Regression linéaire avec sklearn    
model = linear_model.LinearRegression()  
model.fit(X, Y)
YPredicted = model.predict(X)


def colonneX(i):
    t=[]
    
    for j in range(nn):
        t.append(X[j][i])
    return t


X1=np.asarray(colonneX(1)).reshape((nn,1))
X2=np.asarray(colonneX(2)).reshape((nn,1))


from mpl_toolkits.mplot3d import Axes3D
ax=Axes3D(plt.figure())
ax.scatter(X1,X2,Y, c='r')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('DONNEES INITIALES')

from mpl_toolkits.mplot3d import Axes3D
ax2=Axes3D(plt.figure())
ax2.scatter(X1,X2,Y, c='r')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_zlabel('Y')
ax2.set_title('ESTIMATION')
ax2.plot_trisurf(np.reshape(X1,nn), np.reshape(X2, nn), np.reshape(YC, nn))

print()
print('Manuellement : y = {0} +  X1*{1} + X2*{2}'.format(BetaaE[0], BetaaE[1],BetaaE[2]))
print()
print('Sklearn : y = {0} +  X1*{1} + X2*{2}'.format(model.intercept_, model.coef_[0][1],model.coef_[0][2]))



# Partie 4 - On cherche à expliquer l'emission en CO2 avec la masse et la puissance


#Fichier source de l'INSEE
data = pd.read_excel('DataADD.xlsx') # Chemin à changer en fonction de l'emplacement du fichier  


#On définit le nombre de lignes à récupérer dans l'Excel source
nbdata=500


#On enregistres les données dans 3 matrices
def Recupdata():
    puissance=[]
    masse=[]
    co2=[]
    for i in range(0,nbdata):
        puissance.append(data["puissance"][i])
        masse.append(data["masse"][i])
        co2.append(data["co2"][i])
        
    return puissance, masse, co2

puissance,masse,co2 = Recupdata()

puissance = np.asarray(puissance).reshape((nbdata,1))

masse = np.asarray(masse).reshape((nbdata,1))

co2 = np.asarray(co2).reshape((nbdata,1))

#Méthode utilisée pour identifier les valeurs aberrantes (hors de l'intervalle)
def calculDispersion():
    
    m = sorted(masse)
    p = sorted(puissance)
    
    q1M = masse[int(len(masse)*0.25)]
    q3M = masse[int(len(masse)*0.75)]
    ecartM = q3M - q1M

    q1P=puissance[int(len(puissance) * 0.25)]
    q3P=puissance[int(len(puissance) * 0.75)]
    ecartP=q3P - q1P
    
    bInfP=q1P-(1.5*ecartP)      
    bSupP=q3P+(1.5*ecartP)
    
    bInfM=q1M-(1.5*ecartM)
    bSupM=q3M+(1.5*ecartM)
    
    return bInfM,bSupM,bSupP

borneInfM,borneSupMasse,borneSupPuissance = calculDispersion()

print("Q1-1.5*E:" + str(borneInfM))
print("Q3+1.5*E:" + str(borneSupMasse))
print()


#Représentation graphique des données
from mpl_toolkits.mplot3d import Axes3D
ax=Axes3D(plt.figure())
ax.scatter(puissance,masse,co2, c='r')
ax.set_xlabel('Puissance')
ax.set_ylabel('Masse')
ax.set_zlabel('Emissions CO2')
ax.set_title('DONNEES INITIALES')



ax2=Axes3D(plt.figure())
ax2.scatter(puissance,masse,co2, c='r')
ax2.set_xlabel('Puissance')
ax2.set_ylabel('Masse')
ax2.set_zlabel('Emissions CO2')
ax2.set_title('ESTIMATIONS')



#On créé la matrice comprenant les valeurs des variables explicatives Puissance et Masse
def XBis():
    X=np.ones((nbdata,1))
    X=np.concatenate((X,puissance),1)
    X=np.concatenate((X,masse),1)  
    return X



#Rergression linéaire manuelle (forme matricielle)
def estimationBis(YBis):
    Beta= np.dot(np.linalg.inv(np.dot(np.transpose(XBis()),XBis())),np.dot(np.transpose(XBis()),YBis))
    YEstim=np.dot(XBis(),Beta)
    return Beta , YEstim

betabis , ybis = estimationBis(co2)

 
print()  
print("Emission de co2 = " ,betabis[0], " + " ,betabis[1], "* puissance +" , betabis[2], "* masse")
print()


#Module python pour verification des paramètres calculés : R2, betas
import statsmodels.api as sm
est = sm.OLS(co2, XBis()).fit()
print(est.summary())


#On ilustre l'application de la regression linéaire
ax2.plot_trisurf(np.reshape(puissance,nbdata), np.reshape(masse, nbdata), np.reshape(ybis, nbdata))
