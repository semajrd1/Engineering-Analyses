import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.image import NonUniformImage
from matplotlib import rc
import matplotlib.ticker as tck
import math

class Specimen:
    R = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 2.0]]

    def __init__(self, E_1 = 131.0*(10**3), E_2 = 8.274*(10**3), G_12 = 5.033*(10**3), v_12=0.3, S=50.0, X=1.23*(10**3), f=0.4, P_x=4150.0, r=2.38, R_c=2.032, R_t=0.457, t=0.8, phi=[0, 1.5708, 1.5708, 0]):
        self.phi = phi
        self.E_1 = E_1
        self.E_2 = E_2
        self.G_12 = G_12
        self.v_12 = v_12
        self.v_21 = (self.E_2/self.E_1)*self.v_12
        self.m = 1/(1-self.v_12*self.v_21)
        self.f = f
        self.P_x = P_x
        self.r = r
        self.R_c = R_c
        self.R_t = R_t
        self.X = X
        self.S_c = 2.5*S
        self.t = t

        self.theta = np.array(np.zeros(181))
        self.E_theta = np.array(np.zeros(181))
        self.sigma_r = np.array(np.zeros(181))
        self.tau_rtheta = np.array(np.zeros(181))
        self.sigma_theta_1 = np.array(np.zeros(181))
        self.sigma_theta_2 = np.array(np.zeros(181))
        self.sigma_theta_3 = np.array(np.zeros(181))
        self.sigma_theta_4 = np.array(np.zeros(181))
        self.sigma_theta_5 = np.array(np.zeros(181))
        self.sigma_theta = np.array(np.zeros(181))

        self.r_c = np.array(np.zeros(181))
        self.sigma_rc = np.array(np.zeros(181))
        self.sigma_thetac = np.array(np.zeros(181))
        self.tau_rthetac = np.array(np.zeros(181))

        self.sigma_xc = np.array(np.zeros(181))
        self.sigma_yc = np.array(np.zeros(181))
        self.tau_xyc = np.array(np.zeros(181))

        self.epsilon_xc = np.array(np.zeros(181))
        self.epsilon_yc = np.array(np.zeros(181))
        self.gamma_xyc = np.array(np.zeros(181))

        self.epsilon_1c = np.zeros((len(self.phi), len(self.theta)))
        self.epsilon_2c = np.zeros((len(self.phi), len(self.theta)))
        self.gamma_12c = np.zeros((len(self.phi), len(self.theta)))

        self.sigma_xi = np.zeros((len(self.phi), len(self.theta)))
        self.tau_i = np.zeros((len(self.phi), len(self.theta)))
        self.failure_equation = np.zeros((len(self.phi), len(self.theta)))
        self.failure = np.zeros((len(self.phi), len(self.theta)), dtype=object)

        self.hell_A = np.zeros((len(self.phi), len(self.theta)))
        self.hell_B = np.zeros((len(self.phi), len(self.theta)))

    def calculate_abd(self):
        self.S = [[(1/self.E_1), (-self.v_12/self.E_1), 0],
                  [(-self.v_12/self.E_1), (1/self.E_2), 0],
                  [0, 0, (1/self.G_12)]]

        self.Q = np.linalg.inv(self.S)
        self.P_trans = [[math.cos(i) for i in self.phi], [math.sin(i) for i in self.phi]]
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.stresses = np.zeros(6)
        self.strains = np.zeros(6)
        self.Atemp = np.zeros((3, 3))
        self.Btemp = np.zeros((3, 3))
        self.Dtemp = np.zeros((3, 3))
        self.T = np.zeros((len(self.phi), 3, 3))
        self.Q_bar = np.zeros((len(self.phi), 3, 3))
        self.z_values = np.zeros(len(self.phi)+1)

        for i, e in enumerate(self.phi):
            self.T[i] = ([[self.P_trans[0][i]**2, self.P_trans[1][i]**2, -2*self.P_trans[0][i]*self.P_trans[1][i]],
                            [self.P_trans[1][i]**2, self.P_trans[0][i]**2, 2*self.P_trans[0][i]*self.P_trans[1][i]],
                            [self.P_trans[0][i]*self.P_trans[1][i], -self.P_trans[0][i]*self.P_trans[1][i], (self.P_trans[0][i]**2 - self.P_trans[1][i]**2)]])
            self.Q_bar[i] = np.linalg.inv(self.T[i]) @ self.Q @ self.R @ self.T[i] @ np.linalg.inv(self.R)

        #LAMINA THICKNESS
        self.lamina_thickness = self.t/len(self.phi)
        for i in range(len(self.phi)+1):
            self.z_values[i] = (-self.t/2)+self.lamina_thickness*(i)

        #A MATRIX
        for i in range(len(self.phi)):
            self.Atemp = self.Q_bar[i]*(self.z_values[i+1]-self.z_values[i])
            self.A += self.Atemp

        #B MATRIX
        for i in range(len(self.phi)):
            self.Btemp = -(1/2)*self.Q_bar[i]*((self.z_values[i+1])**2-(self.z_values[i])**2)
            self.B += self.Btemp

        #D MATRIX
        for i in range(len(self.phi)):
            self.Dtemp = (1/3)*self.Q_bar[i]*((self.z_values[i+1])**3-(self.z_values[i])**3)
            self.D += self.Dtemp

        #ABD MATRIX
        self.AB = np.concatenate((self.A, self.B), axis=1)
        self.BD = np.concatenate((self.B, self.D), axis=1)
        self.ABD = np.array(np.concatenate((self.AB, self.BD), axis=0))
        self.inv_ABD = np.linalg.inv(self.ABD)

        #LAMINATE MODULI
        self.E_x = (self.ABD[0][0]*self.ABD[1][1]-(self.ABD[0][1]**2))/(self.t*self.ABD[1][1])
        self.E_y = (self.ABD[0][0]*self.ABD[1][1]-(self.ABD[0][1]**2))/(self.t*self.ABD[0][0])
        self.v_xy = self.ABD[0][1]/self.ABD[1][1]
        self.v_yx = (self.v_xy/self.E_x)*self.E_y
        self.G_xy = self.ABD[2][2]/self.t

        # print('E_x (GPa):', self.E_x/1000)
        # print('E_y (GPa):', self.E_y/1000)
        # print('v_xy:', self.v_xy)
        # print('G_xy (GPa):', self.G_xy/1000)

    def ZandU_1984AM(self):
        self.k = np.sqrt(self.E_x/self.E_y)
        self.n = np.sqrt(2*(self.k-self.v_xy)+(self.E_x/self.G_xy))
        self.A_1 = (19*self.n +11*self.n*self.k +10*self.k -10*self.v_xy) +self.f*(11*self.n -6*self.n*self.k +15*self.k -15*self.v_xy)
        self.B_1 = 10*self.n*(1-self.k) +10*self.f*(3*self.k -3*self.v_xy +2*self.n*self.k +self.n)
        self.g = ((1-self.v_xy*self.v_yx)/self.E_y) +(self.k/self.G_xy)

        # this is taken from zhang and ueng and pg9 of Lekhnitskii
        for j in range(len(self.theta)):
            self.theta[j] = np.radians(j-90)
            self.E_theta[j] = 1/((1/self.E_x)*((math.sin(self.theta[j])**4)+((self.n**2)-2*self.k)*(math.sin(self.theta[j])**2) \
                *(math.cos(self.theta[j])**2) +(self.k**2)*(math.cos(self.theta[j])**4)))

            # =========================SIGMA_R==========================
            self.sigma_r[j] = (self.P_x/(2*self.r*self.t))*((2/math.pi)/(2*self.A_1*(self.k -self.v_xy +self.n*self.k)-self.B_1*(self.k -self.v_xy -self.n*self.k))) \
                *((self.B_1*(((3*self.n)/2) -((self.n*self.k)/2) +3*self.k -3*self.v_xy) -self.A_1*(self.n +3*self.n*self.k +4*self.k -4*self.v_xy))*math.cos(self.theta[j]) \
                -(self.B_1*((self.n/2) -((3*self.n*self.k)/2) -3*self.k +3*self.v_xy) -self.A_1*(self.n -self.n*self.k))*math.cos(3*self.theta[j]) \
                -self.B_1*(self.n -self.n*self.k)*math.cos(5*self.theta[j]))

            # =======================TAU_R_THETA========================
            self.tau_rtheta[j] = (self.P_x/(2*self.r*self.t))*((2/math.pi)/(2*self.A_1*(self.k -self.v_xy +self.n*self.k)-self.B_1*(self.k -self.v_xy -self.n*self.k))) \
                *((self.A_1*(self.n -self.n*self.k) +self.B_1*(((-3*self.n*self.k)/2) -((3*self.n)/2) -self.k +self.v_xy))*math.sin(self.theta[j]) \
                -(self.B_1*(((self.n*self.k)/2) +((5*self.n)/2) +self.k -self.v_xy) -self.A_1*(self.n -self.n*self.k))*math.sin(3*self.theta[j]) -self.B_1 \
                *(self.n -self.n*self.k)*math.sin(5*self.theta[j]))

            # =======================SIGMA_THETA========================
            self.sigma_theta_1[j] = ((2*self.E_theta[j])/(math.pi*self.E_x))*(-self.v_xy*(math.cos(self.theta[j])**4) \
                -(self.k**2 -1 +2*self.v_xy)*(math.sin(self.theta[j])**2)*(math.cos(self.theta[j])**2) \
                +(2 +2*self.k -self.v_xy -(self.n**2))*(math.sin(self.theta[j])**4))*math.cos(self.theta[j])
            self.sigma_theta_2[j] = (((2*self.E_theta[j]*(2*self.A_1 -self.B_1))/(math.pi*self.E_x))/(2*self.A_1*(self.k -self.v_xy +self.n*self.k) -self.B_1*(self.k -self.v_xy -self.n*self.k))) \
                *(((self.n/2)*math.cos(2*self.theta[j]) -((math.sin(self.theta[j])**2)-self.k*(math.cos(self.theta[j])**2)))*((1+2*self.k)*(self.k -self.v_xy +self.n)*(math.cos(self.theta[j])**2) \
                -(self.k*self.v_xy +(self.n**2)*self.k -(self.k**2) -self.v_xy*self.n)*(math.sin(self.theta[j])**2))*math.sin(2*self.theta[j])*math.sin(self.theta[j]) \
                +(((math.sin(self.theta[j])**2)-self.k*(math.cos(self.theta[j])**2))*math.cos(2*self.theta[j]) +(self.n/2)*(math.sin(2*self.theta[j])**2))*((-(self.n**2) +self.k -self.v_xy \
                +self.n*self.v_xy)*(math.cos(self.theta[j])**2) +(2+self.k)*(self.n*self.k +self.k -self.v_xy)*(math.sin(self.theta[j])**2))*math.cos(self.theta[j]))
            self.sigma_theta_3[j] = (((-4*self.E_theta[j]*self.B_1)/(math.pi*self.E_x))/(2*self.A_1*(self.k -self.v_xy +self.n*self.k) -self.B_1*(self.k -self.v_xy -self.n*self.k))) \
                *(((self.n/2)*math.cos(2*self.theta[j]) -((math.sin(self.theta[j])**2) -self.k*(math.cos(self.theta[j])**2)))*(self.n*(1+2*self.k)*(math.cos(self.theta[j])**2) \
                +self.n*self.v_xy*(math.sin(self.theta[j])**2))*math.sin(2*self.theta[j])*math.sin(self.theta[j]) +(((math.sin(self.theta[j])**2)-self.k*(math.cos(self.theta[j])**2)) \
                *math.cos(2*self.theta[j]) +(self.n/2)*(math.sin(2*self.theta[j])**2))*((-(self.n**2) +self.k -self.v_xy)*(math.cos(self.theta[j])**2) +(self.k-self.v_xy)*(2+self.k) \
                *(math.sin(self.theta[j])**2))*math.cos(self.theta[j]))
            self.sigma_theta_4[j] = (((-4*self.E_theta[j]*self.B_1)/(math.pi*self.E_x))/(2*self.A_1*(self.k -self.v_xy +self.n*self.k)-self.B_1*(self.k -self.v_xy -self.n*self.k))) \
                *(((self.n/2)*math.cos(4*self.theta[j])-2*((math.sin(self.theta[j])**2) -self.k*(math.cos(self.theta[j])**2))*math.cos(2*self.theta[j])) \
                *((1+2*self.k)*(self.k -self.v_xy +self.n)*(math.cos(self.theta[j])**2) -(self.k*self.v_xy +(self.n**2)*self.k -(self.k**2) -self.v_xy*self.n)*(math.sin(self.theta[j])**2)) \
                *math.sin(2*self.theta[j])*math.sin(self.theta[j]) +(((math.sin(self.theta[j])**2)-self.k*(math.cos(self.theta[j])**2))*math.cos(4*self.theta[j]) \
                +(self.n/2)*math.sin(4*self.theta[j])*math.sin(2*self.theta[j]))*((self.k -self.v_xy +self.n*self.v_xy -(self.n**2))*(math.cos(self.theta[j])**2) \
                +(2 +self.k)*(self.n*self.k + self.k -self.v_xy)*(math.sin(self.theta[j])**2))*math.cos(self.theta[j]))
            self.sigma_theta_5[j] = (((self.g*self.E_theta[j]*(2*self.A_1 -self.B_1))/math.pi)/(2*self.A_1*(self.k -self.v_xy \
                +self.n*self.k) -self.B_1*(self.k -self.v_xy -self.n*self.k)))*(math.sin(self.theta[j])**2)
            # -----------------------------------------------------------
            self.sigma_theta[j] = (self.P_x/(2*self.r*self.t))*(self.sigma_theta_1[j] + self.sigma_theta_2[j] + self.sigma_theta_3[j] + self.sigma_theta_4[j] + self.sigma_theta_5[j])

    def yamada_sun(self):
        for k in range(len(self.phi)):
            for j in range(len(self.theta)):

            # =======================JD METHOD========================    
                self.r_c[j]= self.r +self.R_t +(self.R_c -self.R_t)*math.cos(self.theta[j])

                self.sigma_rc[j] = (self.r/self.r_c[j])*self.sigma_r[j]
                self.sigma_thetac[j] = (self.r/self.r_c[j])*self.sigma_theta[j]
                self.tau_rthetac[j] = (self.r/self.r_c[j])*self.tau_rtheta[j]

                self.sigma_xc[j] = ((self.sigma_rc[j]+self.sigma_thetac[j])/2) +((self.sigma_rc[j]-self.sigma_thetac[j])/2)*math.cos(-2*self.theta[j]) +self.tau_rthetac[j]*math.sin(-2*self.theta[j])
                self.sigma_yc[j] = ((self.sigma_rc[j]+self.sigma_thetac[j])/2) -((self.sigma_rc[j]-self.sigma_thetac[j])/2)*math.cos(-2*self.theta[j]) -self.tau_rthetac[j]*math.sin(-2*self.theta[j])
                self.tau_xyc[j] = -((self.sigma_rc[j]-self.sigma_thetac[j])/2)*math.sin(-2*self.theta[j]) +self.tau_rthetac[j]*math.cos(-2*self.theta[j])

                # obtained from composite compliance matrix
                self.epsilon_xc[j] = (self.sigma_xc[j]/self.E_x) -self.sigma_yc[j]*(self.v_yx/self.E_y)
                self.epsilon_yc[j] = (self.sigma_yc[j]/self.E_y) -self.sigma_xc[j]*(self.v_xy/self.E_x)
                self.gamma_xyc[j] = self.tau_xyc[j]/self.G_xy

                self.epsilon_1c[k,j] = ((self.epsilon_xc[j] +self.epsilon_yc[j])/2) +((self.epsilon_xc[j] -self.epsilon_yc[j])/2)*math.cos(2*self.phi[k]) +(self.gamma_xyc[j]/2)*math.sin(2*self.phi[k])
                self.epsilon_2c[k,j] = ((self.epsilon_xc[j] +self.epsilon_yc[j])/2) -((self.epsilon_xc[j] -self.epsilon_yc[j])/2)*math.cos(2*self.phi[k]) -(self.gamma_xyc[j]/2)*math.sin(2*self.phi[k])
                self.gamma_12c[k,j] = -((self.epsilon_xc[j] -self.epsilon_yc[j])/2)*math.sin(2*self.phi[k]) +(self.gamma_xyc[j]/2)*math.cos(2*self.phi[k])

                self.sigma_xi[k,j] = self.m*self.E_1*(self.epsilon_1c[k,j] +self.v_21*self.epsilon_2c[k,j])
                self.tau_i[k,j] = self.G_12*self.gamma_12c[k,j]

                self.failure_equation[k,j] = ((self.sigma_xi[k,j]/self.X)**2)+((self.tau_i[k,j]/self.S_c)**2)
                if self.failure_equation[k,j] < 1:
                    self.failure[k,j] = 0
                else:
                    self.failure[k,j] = 1
        
def main():
    A1 = np.array(np.zeros(6, dtype=object))
    for i in range(len(A1)):
        f = 0.4
        phi=[0,90,90,0]
        for j in range(len(phi)):
            phi[j]=math.radians(phi[j])
        A1[i]=Specimen(f=f, phi=phi)
        A1[i].calculate_abd()
        A1[i].ZandU_1984AM()
        A1[i].yamada_sun()

    # FOR PSO
    # for i in range(len(A1)):
    #     f = i/(len(A1)-1)
    #     print('=============================')
    #     print('friction coef. :',f)
    #     A1[i]=compBolt(f=f, phi=z, t=thickness)
    #     A1[i].calculate_abd()
    #     A1[i].ZandU_1984AM()
    #     A1[i].yamada_sun()

# ====================================
# ====================================

    ply_number = 0

    rc('font',**{'family':'sans-serif','sans-serif':['Courier']})
    rc('text', usetex=False)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(4, 7))
    x = np.degrees(A1[0].theta)

# ====================================
    y1_2 = A1[0].sigma_r
    y2_2 = A1[1].sigma_r
    y3_2 = A1[2].sigma_r
    y4_2 = A1[3].sigma_r
    y5_2 = A1[4].sigma_r
    y6_2 = A1[5].sigma_r

    ax1.plot(x, y1_2, label='f=0.0')
    ax1.plot(x, y2_2, label='f=0.2')
    ax1.plot(x, y3_2, label='f=0.4')
    ax1.plot(x, y4_2, label='f=0.6')
    ax1.plot(x, y5_2, label='f=0.8')
    ax1.plot(x, y6_2, label='f=1.0')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlim(-90, 90)
    ax1.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax1.set_ylabel(r'$\sigma_r$'+' /MPa')
    ax1.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax1.set_title('Graph of '+r'$\sigma_{r}$'+'\n Distribution vs. Angle '+r'$(\phi)$')  
    ax1.grid()
# ====================================
    y1_3 = A1[0].sigma_theta
    y2_3 = A1[1].sigma_theta
    y3_3 = A1[2].sigma_theta
    y4_3 = A1[3].sigma_theta
    y5_3 = A1[4].sigma_theta
    y6_3 = A1[5].sigma_theta

    ax2.plot(x, y1_3, label='f=0.0')
    ax2.plot(x, y2_3, label='f=0.2')
    ax2.plot(x, y3_3, label='f=0.4')
    ax2.plot(x, y4_3, label='f=0.6')
    ax2.plot(x, y5_3, label='f=0.8')
    ax2.plot(x, y6_3, label='f=1.0')

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim(-90, 90)
    ax2.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax2.set_ylabel(r'$\sigma_{\theta}$'+' /MPa')
    ax2.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax2.set_title('Graph of '+r'$\sigma_{\theta}$'+'\n Distribution vs. Angle '+r'$(\phi)$')  
    ax2.grid()
# =======================================
    y1_1 = A1[0].tau_rtheta
    y2_1 = A1[1].tau_rtheta
    y3_1 = A1[2].tau_rtheta
    y4_1 = A1[3].tau_rtheta
    y5_1 = A1[4].tau_rtheta
    y6_1 = A1[5].tau_rtheta

    ax3.plot(x, y1_1, label='f=0.0')
    ax3.plot(x, y2_1, label='f=0.2')
    ax3.plot(x, y3_1, label='f=0.4')
    ax3.plot(x, y4_1, label='f=0.6')
    ax3.plot(x, y5_1, label='f=0.8')
    ax3.plot(x, y6_1, label='f=1.0')

    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_xlim(-90, 90)
    ax3.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax3.set_ylabel(r'$\tau_{r\theta}$'+' /MPa')
    ax3.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax3.set_title('Graph of '+r'$\tau_{r\theta}$'+'\n Distribution vs. Angle '+r'$(\phi)$')
    ax3.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

# ====================================

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(4, 7))
    x = np.degrees(A1[0].theta)

# =======================================
    y1_1 = A1[0].sigma_xc
    y2_1 = A1[1].sigma_xc
    y3_1 = A1[2].sigma_xc
    y4_1 = A1[3].sigma_xc
    y5_1 = A1[4].sigma_xc
    y6_1 = A1[5].sigma_xc

    ax1.plot(x, y1_1, label='f=0.0')
    ax1.plot(x, y2_1, label='f=0.2')
    ax1.plot(x, y3_1, label='f=0.4')
    ax1.plot(x, y4_1, label='f=0.6')
    ax1.plot(x, y5_1, label='f=0.8')
    ax1.plot(x, y6_1, label='f=1.0')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlim(-90, 90)
    ax1.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax1.set_ylabel(r'$\sigma_{xc}$'+' /MPa')
    ax1.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax1.set_title('Graph of '+r'$\sigma_{xc}$'+' Distribution vs. Angle '+r'$(\phi)$')
    ax1.grid()
# ====================================
    y1_2 = A1[0].sigma_yc
    y2_2 = A1[1].sigma_yc
    y3_2 = A1[2].sigma_yc
    y4_2 = A1[3].sigma_yc
    y5_2 = A1[4].sigma_yc
    y6_2 = A1[5].sigma_yc

    ax2.plot(x, y1_2, label='f=0.0')
    ax2.plot(x, y2_2, label='f=0.2')
    ax2.plot(x, y3_2, label='f=0.4')
    ax2.plot(x, y4_2, label='f=0.6')
    ax2.plot(x, y5_2, label='f=0.8')
    ax2.plot(x, y6_2, label='f=1.0')

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim(-90, 90)
    ax2.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax2.set_ylabel(r'$\sigma_{yc}$'+' /MPa')
    ax2.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax2.set_title('Graph of '+r'$\sigma_{yc}$'+' Distribution vs. Angle '+r'$(\phi)$')  
    ax2.grid()
# ====================================
    y1_3 = A1[0].tau_xyc
    y2_3 = A1[1].tau_xyc
    y3_3 = A1[2].tau_xyc
    y4_3 = A1[3].tau_xyc
    y5_3 = A1[4].tau_xyc
    y6_3 = A1[5].tau_xyc

    ax3.plot(x, y1_3, label='f=0.0')
    ax3.plot(x, y2_3, label='f=0.2')
    ax3.plot(x, y3_3, label='f=0.4')
    ax3.plot(x, y4_3, label='f=0.6')
    ax3.plot(x, y5_3, label='f=0.8')
    ax3.plot(x, y6_3, label='f=1.0')

    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_xlim(-90, 90)
    ax3.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax3.set_ylabel(r'$\tau_{xyc}$'+' /MPa')
    ax3.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax3.set_title('Graph of '+r'$\tau_{xyc}$'+' Distribution vs. Angle '+r'$(\phi)$')  
    ax3.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

# =======================================

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(4, 7))
    x = np.degrees(A1[0].theta)

# =======================================
    y1_1 = A1[0].epsilon_xc
    y2_1 = A1[1].epsilon_xc
    y3_1 = A1[2].epsilon_xc
    y4_1 = A1[3].epsilon_xc
    y5_1 = A1[4].epsilon_xc
    y6_1 = A1[5].epsilon_xc

    ax1.plot(x, y1_1, label='f=0.0')
    ax1.plot(x, y2_1, label='f=0.2')
    ax1.plot(x, y3_1, label='f=0.4')
    ax1.plot(x, y4_1, label='f=0.6')
    ax1.plot(x, y5_1, label='f=0.8')
    ax1.plot(x, y6_1, label='f=1.0')

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlim(-90, 90)
    ax1.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax1.set_ylabel(r'$\epsilon_{xc}$'+' /MPa')
    ax1.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax1.set_title('Graph of '+r'$\epsilon_{xc}$'+' Distribution vs. Angle '+r'$(\phi)$')
    ax1.grid()
# ====================================
    y1_2 = A1[0].epsilon_yc
    y2_2 = A1[1].epsilon_yc
    y3_2 = A1[2].epsilon_yc
    y4_2 = A1[3].epsilon_yc
    y5_2 = A1[4].epsilon_yc
    y6_2 = A1[5].epsilon_yc

    ax2.plot(x, y1_2, label='f=0.0')
    ax2.plot(x, y2_2, label='f=0.2')
    ax2.plot(x, y3_2, label='f=0.4')
    ax2.plot(x, y4_2, label='f=0.6')
    ax2.plot(x, y5_2, label='f=0.8')
    ax2.plot(x, y6_2, label='f=1.0')

    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlim(-90, 90)
    ax2.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax2.set_ylabel(r'$\epsilon_{yc}$'+' /MPa')
    ax2.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax2.set_title('Graph of '+r'$\epsilon_{yc}$'+' Distribution vs. Angle '+r'$(\phi)$')  
    ax2.grid()
# ====================================
    y1_3 = A1[0].gamma_xyc
    y2_3 = A1[1].gamma_xyc
    y3_3 = A1[2].gamma_xyc
    y4_3 = A1[3].gamma_xyc
    y5_3 = A1[4].gamma_xyc
    y6_3 = A1[5].gamma_xyc

    ax3.plot(x, y1_3, label='f=0.0')
    ax3.plot(x, y2_3, label='f=0.2')
    ax3.plot(x, y3_3, label='f=0.4')
    ax3.plot(x, y4_3, label='f=0.6')
    ax3.plot(x, y5_3, label='f=0.8')
    ax3.plot(x, y6_3, label='f=1.0')

    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_xlim(-90, 90)
    ax3.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax3.set_ylabel(r'$\gamma_{xyc}$'+' /MPa')
    ax3.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax3.set_title('Graph of '+r'$\gamma_{xyc}$'+' Distribution vs. Angle '+r'$(\phi)$')  
    ax3.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

# ====================================

    fig, ax = plt.subplots()
    y1 = A1[0].failure_equation[ply_number]
    y2 = A1[1].failure_equation[ply_number]
    y3 = A1[2].failure_equation[ply_number]
    y4 = A1[3].failure_equation[ply_number]
    y5 = A1[4].failure_equation[ply_number]
    y6 = A1[5].failure_equation[ply_number]

    ax.plot(x, y1, label='f=0.0')
    ax.plot(x, y2, label='f=0.2')
    ax.plot(x, y3, label='f=0.4')
    ax.plot(x, y4, label='f=0.6')
    ax.plot(x, y5, label='f=0.8')
    ax.plot(x, y6, label='f=1.0')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax.set_ylabel('Failure Index')
    ax.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax.set_title('Ply Number '+str(ply_number+1)+' '+'('+str(np.degrees(A1[0].phi[ply_number]))+r'$^\circ$'+')'+':\n Graph of Failure Index Distribution vs. Angle '+r'$(\phi)$')  
    ax.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

    # ====================================

    fig, ax = plt.subplots()
    y1 = A1[0].failure_equation[1]
    y2 = A1[1].failure_equation[1]
    y3 = A1[2].failure_equation[1]
    y4 = A1[3].failure_equation[1]
    y5 = A1[4].failure_equation[1]
    y6 = A1[5].failure_equation[1]

    ax.plot(x, y1, label='f=0.0')
    ax.plot(x, y2, label='f=0.2')
    ax.plot(x, y3, label='f=0.4')
    ax.plot(x, y4, label='f=0.6')
    ax.plot(x, y5, label='f=0.8')
    ax.plot(x, y6, label='f=1.0')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax.set_ylabel('Failure Index')
    ax.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax.set_title('Ply Number '+str(2)+' '+'('+str(np.degrees(A1[0].phi[1]))+r'$^\circ$'+')'+':\n Graph of Failure Index Distribution vs. Angle '+r'$(\phi)$')  
    ax.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

    # ====================================

    fig, ax = plt.subplots()
    y1 = A1[0].failure_equation[2]
    y2 = A1[1].failure_equation[2]
    y3 = A1[2].failure_equation[2]
    y4 = A1[3].failure_equation[2]
    y5 = A1[4].failure_equation[2]
    y6 = A1[5].failure_equation[2]

    ax.plot(x, y1, label='f=0.0')
    ax.plot(x, y2, label='f=0.2')
    ax.plot(x, y3, label='f=0.4')
    ax.plot(x, y4, label='f=0.6')
    ax.plot(x, y5, label='f=0.8')
    ax.plot(x, y6, label='f=1.0')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax.set_ylabel('Failure Index')
    ax.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax.set_title('Ply Number '+str(3)+' '+'('+str(np.degrees(A1[0].phi[2]))+r'$^\circ$'+')'+':\n Graph of Failure Index Distribution vs. Angle '+r'$(\phi)$')  
    ax.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)

    # ====================================

    fig, ax = plt.subplots()
    y1 = A1[0].failure_equation[3]
    y2 = A1[1].failure_equation[3]
    y3 = A1[2].failure_equation[3]
    y4 = A1[3].failure_equation[3]
    y5 = A1[4].failure_equation[3]
    y6 = A1[5].failure_equation[3]

    ax.plot(x, y1, label='f=0.0')
    ax.plot(x, y2, label='f=0.2')
    ax.plot(x, y3, label='f=0.4')
    ax.plot(x, y4, label='f=0.6')
    ax.plot(x, y5, label='f=0.8')
    ax.plot(x, y6, label='f=1.0')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(-90, 90)
    ax.set_xticks(np.arange(min(x), max(x)+30, 30))
    ax.set_ylabel('Failure Index')
    ax.set_xlabel(r'$\phi$'+' /degrees '+r'$(^\circ)$')
    ax.set_title('Ply Number '+str(4)+' '+'('+str(np.degrees(A1[0].phi[3]))+r'$^\circ$'+')'+':\n Graph of Failure Index Distribution vs. Angle '+r'$(\phi)$')  
    ax.grid()

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=1.0)
    plt.show()

if __name__ == '__main__':
    main()

