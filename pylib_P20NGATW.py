# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:54:16 2022

@author: PVB
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:32:26 2021

@author: ben
"""

import numpy as np

class Phung2020h_NGATW:
    def __init__(self):
        self.periods = np.array([0, 0.01, 0.02, 0.03,  0.04, 0.05,  0.075,  0.1,  0.12,  0.15,  0.17,  0.2,  0.25, 0.30, 0.4,  0.5, 0.75,  1.0,   1.5,   2,  3,  4,  5,  7.5,  10])

        self.c4 = np.array([-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1,-2.1])
        self.c4_a = np.array([-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5])
        self.c_RB = np.array([50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50])
        self.c2 = np.array([1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06,1.06])

        self.c1 = np.array([-1.4526,	-1.4468,	-1.4066,	-1.3175,	-1.1970,	-1.0642,	-0.7737,	-0.5958,	-0.5229,	-0.5005,	-0.5165,	-0.5558,	-0.6550,	-0.7898,	-1.0839,	-1.3279,	-1.9346,	-2.3932,	-2.9412,	-3.2794,	-3.5891,	-3.8483,	-3.9458,	-4.2514,	-4.5075])

        self.c3 = np.array([1.4379,1.4379,1.4030,1.3628,1.3168,1.2552,1.1873,1.2485,1.3263,1.4098,1.4504,1.5274,1.6737,1.8298,2.0330,2.2044,2.4664,2.6194,2.7708,2.8699,2.9293,3.0012,3.0012,3.0012,3.0012])
        self.c_n = np.array([12.14866822,12.14866822,12.24803407,12.53378414,12.99189704,13.65075358,15.71447541,16.77262242,16.77563032,16.18679785,15.84314399,15.01467129,12.69643148,10.44981091,6.802216656,4.41069375,3.4064,3.1612,2.8078,2.4631,2.2111,1.9668,1.6671,1.5737,1.5265])
        self.c_m = np.array([5.50455,5.51303,5.51745,5.51798,5.51462,5.50692,5.43078,5.42081,5.46158,5.55373,5.60449,5.64383,5.66058,5.65301,5.62843,5.59326,5.56641,5.60836,5.73551,5.85199,6.08195,6.25683,6.39882,6.66908,6.84353])

        self.c_g1= np.array([-0.0087980,-0.0087980,-0.0090670,-0.0094510,-0.0098320,-0.0101940,-0.0109620,-0.0114520,-0.0115970,-0.0115790,-0.0113030,-0.0108190,-0.0100190,-0.0092670,-0.0079030,-0.0069990,-0.0054380,-0.0045400,-0.0036370,-0.0029726,-0.0024872,-0.0021234,-0.0017638,-0.0010788,-0.0007423])
        self.c_g2= np.array([-0.007127092,-0.007127092,-0.007248737,-0.007327856,-0.007361759,-0.007360913,-0.007051574,-0.005719182,-0.00436511,-0.002649555,-0.001999512,-0.001254506,-0.00075041,-0.000447155,-0.000247246,-0.000416797,-0.001131462,-0.001741492,-0.002427965,-0.002705545,-0.004107346,-0.005776395,-0.007747849,-0.009141588,-0.012633296])
        self.c_g3= np.array([4.225634814,4.225634814,4.230341898,4.236182109,4.250188668,4.303122568,4.446126947,4.610835161,4.723496543,4.878140618,4.981707053,5.066410859,5.21986472,5.32821979,5.201761713,5.187931728,4.877209058,4.63975087,4.571203643,4.425116502,3.6219035,3.48626393,3.277906342,3.074948086,3.074948086])

        self.c5 = np.array([6.4551,6.4551,6.4551,6.4551,6.4551,6.4551,6.4551,6.8305,7.1333,7.3621,7.4365,7.4972,7.5416,7.5600,7.5735,7.5778,7.5808,7.5814,7.5817,7.5818,7.5818,7.5818,7.5818,7.5818,7.5818])
        self.c_HM = np.array([3.0956,3.0956,3.0963,3.0974,3.0988,3.1011,3.1094,3.2381,3.3407,3.4300,3.4688,3.5146,3.5746,3.6232,3.6945,3.7401,3.7941,3.8144,3.8284,3.8330,3.8361,3.8369,3.8376,3.8380,3.8380])
        self.c6 = np.array([0.4908,0.4908,0.4925,0.4992,0.5037,0.5048,0.5048,0.5048,0.5048,0.5045,0.5036,0.5016,0.4971,0.4919,0.4807,0.4707,0.4575,0.4522,0.4501,0.4500,0.4500,0.4500,0.4500,0.4500,0.4500])
        self.dp = np.array([-6.785205271,-6.750647967,-6.716179208,-6.681798923,-6.647507037,-6.613303475,-6.52818049,-6.443607867,-6.376345237,-6.276109027,-6.209722535,-6.110797854,-5.947665543,-5.786703295,-5.47125339,-5.164376146,-4.4342041,-3.75596604,-2.55026692,-1.536847647,-0.052837841,0,0,0,0])

        self.phi2= np.array([-0.1417,-0.1417,-0.1364,-0.1403,-0.1591,-0.1862,-0.2538,-0.2943,-0.3077,-0.3113,-0.3062,-0.2927,-0.2662,-0.2405,-0.1975,-0.1633,-0.1028,-0.0699,-0.0425,-0.0302,-0.0129,-0.0016,0.0000,0.0000,0.0000])
        self.phi3= np.array([-0.007010, -0.007010,-0.007279,-0.007354,-0.006977,-0.006467,-0.005734,-0.005604,-0.005696,-0.005845,-0.005959,-0.006141,-0.006439,-0.006704,-0.007125,-0.007435,-0.008120,-0.008444,-0.007707,-0.004792,-0.001828,-0.001523,-0.001440,-0.001369,-0.001361])
        self.phi4= np.array([0.102151,0.102151,0.108360,0.119888,0.133641,0.148927,0.190596,0.230662,0.253169,0.266468,0.265060,0.255253,0.231541,0.207277,0.165464,0.133828,0.085153,0.058595,0.031787,0.019716,0.009643,0.005379,0.003223,0.001134,0.000515])                                                                                                                      

        self.c7= np.array([0.00803536,0.00803536,0.007592927,0.007250488,0.007006048,0.006860143,0.007007726,0.007246641,0.007455965,0.00770271,0.007798775,0.007823121,0.00807121,0.008395901,0.00927498,0.010165826,0.012793392,0.013761922,0.013899933,0.012559337,0.009183764,0.004796976,0.001067909,-0.004234005,-0.006203311])
        self.c7_b= np.array([0.021034339,0.021034339,0.021638743,0.022052403,0.022283866,0.022340988,0.021712418,0.020031223,0.018584674,0.016544376,0.015412673,0.014410752,0.013237789,0.011957864,0.00946882,0.005799966,-0.003683309,-0.008131001,-0.010287269,-0.008563294,-0.003058727,0.003919649,0.013063958,0.027920315,0.04195315])

        self.c1_a= np.array([0.137929376,0.131008944,0.124713602,0.119040284,0.113983621,0.109535952,0.101016047,0.096066418,0.094563811,0.096331606,0.100411816,0.113754448,0.132878713,0.147312358,0.158162078,0.163112167,0.169389333,0.177254643,0.17612706,0.161185738,0.112720638,0.053953075,0.053953075,0.053953075,0.053953075])
        self.c1_c= np.array([0.04272907,0.05491546,0.06599634,0.075907583,0.08465752,0.092384148,0.109529939,0.11002146,0.101539072,0.081268087,0.066332395,0.047001724,0.018708157, 0,0,0,0,0,0,0,0,0,0,0,0])

        self.c1_b = np.array([0,0,0,0,0,0,0,0,0,0,0,0,-0.034932397,-0.052793111,-0.096705093,-0.121161057,-0.158672494,-0.184203622,-0.218917854,-0.218956446,-0.218956446,-0.218956446,-0.218956446,-0.218956446,-0.218956446])
        self.c1_d = np.array([-0.165254064,-0.164615502,-0.166706035,-0.19413385,-0.2133523,-0.246430796,-0.240863766,-0.22991286,-0.171017133,-0.13673324,-0.085084587,-0.078934463,-2.86E-07,0,0,0,0,0,0,0,0,0,0,0,0])

        self.c11= np.array([-0.108037007,-0.108037007,-0.102071888,-0.104638092,-0.105159212,-0.09694663,-0.079174009,-0.120806584,-0.127655488,-0.123958373,-0.120234904,-0.128554524,-0.104990465,-0.125335213,-0.131458922,-0.102606613,-0.072842909,-0.072286657,-0.143270261,-0.171095562,-0.269171794,-0.321537372,-0.344321787,-0.379466889,-0.478010668])
        self.c11_b = np.array([0.195951708,0.195951708,0.181778172,0.163170085,0.142063237,0.098053885,0.046296818,0.173997245,0.209294889,0.217339629,0.218818569,0.262936287,0.231024464,0.27034386,0.306056087,0.272617073,0.265158493,0.303895403,0.443286099,0.520454201,0.817526599,1.015932218,0.892205391,0.86436398,1.443597529])

        self.phi1TW	= np.array([-0.510745033,	-0.510415026,	-0.502941955,	-0.491366306,	-0.474484696,	-0.459984157,	-0.446396645,	-0.476282069,	-0.4931516,	-0.517925624,	-0.532965478,	-0.547665313,	-0.565549294,	-0.606451856,	-0.653316566,	-0.674933921,	-0.796961941,	-0.884871551,	-0.958271065,	-0.968084348,	-0.96759396,	-0.964753341,	-0.923270348,	-0.85471647,	-0.770092758])
        self.phi5TW	= np.array([0.07436,	0.07436,	0.07359,	0.07713,	0.08249,	0.09010,	0.10291,	0.12596,	0.11942,	0.10019,	0.08862,	0.08048,	0.08000,	0.08013,	0.07916,	0.07543,	0.07573,	0.07941,	0.12820,	0.16687,	0.20292,	0.17899,	0.17368,	0.15176,	0.14062])

 
        self.tau = np.array([0.4913,0.4924,0.4954,0.4935,0.4919,0.4922,0.4927,0.4976,0.4942,0.4992,0.4955,0.4968,0.4966,0.4998,0.5173,0.5427,0.5618,0.5902,0.5829,0.5476,0.4889,0.5195,0.5151,0.4978,0.5369])
        self.PhiS2S = np.array([0.3002,0.3003,0.2959,0.3060,0.3325,0.3638,0.3997,0.3993,0.3870,0.3538,0.3410,0.3282,0.2917,0.2906,0.3094,0.3276,0.3455,0.2682,0.2634,0.2745,0.2790,0.2663,0.2385,0.1916,0.1324])
        self.PhiSS = np.array([0.3125,0.3134,0.3174,0.3292,0.3406,0.3521,0.3397,0.3125,0.2999,0.2937,0.3001,0.3061,0.3077,0.3098,0.3151,0.3100,0.3198,0.3254,0.3308,0.3388,0.3277,0.2939,0.2716,0.2393,0.2412])
        self.SigT = np.array([0.6551,0.6564,0.6586,0.6675,0.6845,0.7061,0.7197,0.7104,0.6956,0.6787,0.6721,0.6695,0.6530,0.6559,0.6801,0.7057,0.7330,0.7253,0.7201,0.7000,0.6513,0.6536,0.6292,0.5846,0.6033])


    def fmag(self, mag):
        term6 = self.c2*(mag-6)

        term7 = (self.c2 - self.c3)/self.c_n*np.log(1 + np.exp(self.c_n*(self.c_m - mag)))

        f_mag = term6 + term7
        return f_mag

    def fsource(self, mag, ztor, rake, delta):

        if (rake >= 30) & (rake <= 150):
            F_RV = 1
            F_NM = 0
 
        elif (rake >= -120) & (rake <= -60):
            F_RV = 0
            F_NM = 1
      
        else:
            F_RV = 0
            F_NM = 0
      

        if F_RV == 1:
            E_ztor = (np.max(3.5384-2.60*np.max(mag-5.8530,0),0))**2
        else:
            E_ztor = (np.max(2.7482-1.7639*np.max(mag-5.5210,0),0))**2

        if ztor == 999:
            del_ztor = 0
        else:
            del_ztor = ztor - E_ztor
        # ztor term    
        term4 = (self.c7+self.c7_b/np.cosh(2*np.max(mag-4.5,0)))*del_ztor

        # Sof term

        term2 = (self.c1_a + self.c1_c/np.cosh(2*np.max(mag-4.5,0)))*F_RV
        term3 = (self.c1_b + self.c1_d/np.cosh(2*np.max(mag-4.5,0)))*F_NM

        # dip term 
        delta = delta*np.pi/180
        term5 = (self.c11 + self.c11_b/np.cosh(2*np.max(mag-4.5,0)))*(np.cos(delta)**2)

        f_source = term2 + term3 + term4 + term5

        return f_source

    def fpath(self, mag, ztor, Rrup):

        if (ztor>20) & (mag<7):
            term8 =  self.c4*np.log(Rrup + (self.c5 + self.dp*np.max(ztor/50-20/50,0))*np.cosh(self.c6*np.max(mag-self.c_HM,0)))
        else:

            term8 =  self.c4*np.log(Rrup + self.c5*np.cosh(self.c6*np.max(mag-self.c_HM,0)))


    
        term9 =  (self.c4_a - self.c4)*np.log(np.sqrt(Rrup**2 + self.c_RB**2))

        term10 = (self.c_g1 + self.c_g2/(np.cosh(np.max(mag-self.c_g3,0))))*Rrup

        f_path = term8 + term9 + term10

        return f_path


    def Ph2020_TW_all_per(self, mag, ztor, rake, delta, Rrup, Vs30, Z1):
        # predited ground motion at rock
        Sa1130 = np.exp(self.c1 + self.fmag(mag) +
                 self.fsource(mag, ztor, rake, delta) +
                 self.fpath(mag, ztor, Rrup)) 

        # linear term

        term14 = self.phi1TW*min(np.log(Vs30/1130),0)
        
        # Non-linear term
        term15 = self.phi2*(np.exp(self.phi3*(min(Vs30,1130)-360))-np.exp(self.phi3*(1130-360)))*np.log((Sa1130 + self.phi4)/self.phi4)


        # basin effect
        Ez_1 = np.exp(-3.9/2*np.log((Vs30**2 + 262**2)/(1700**2 + 262**2)))


        if Z1 == -999:
            term16 = 0
        else:
            del_Z1 = Z1 - Ez_1
            term16 = self.phi5TW*(1 - np.exp(-del_Z1/300))

        # predicted ground motion at the surface    
        Sa = Sa1130*np.exp(term14 + term15 + term16)

        # standard deviations
        Sig_T = self.SigT

        Sig_SS = np.sqrt(self.tau**2 + self.PhiSS**2)

        return (self.periods, Sa, Sig_T, Sig_SS)

    def Ph2020_TW_p(self, periods, mag, ztor, rake, delta, Rrup, Vs30, Z1):
        
        #convert periods to numpy array
        periods = np.array([periods]).flatten()
        
        #period tolerance
        p_tol = 1e-4
        #compute conditional psa for all periods
        per_all, Sa_all, sigT_all, sigS_all = self.Ph2020_TW_all_per(mag, ztor, rake, delta, Rrup, Vs30, Z1)

        #find eas for frequency of interest
        if np.all([np.isclose(p, per_all, p_tol).any() for p in periods]):
            i_p   = np.array([np.where(np.isclose(p, per_all, p_tol))[0] for p in periods]).flatten()
            Sa   = Sa_all[i_p]
            SigmaT = sigT_all[i_p]
            SigmaS = sigS_all[i_p]
       
        else:
            Sa   = np.exp(np.interp(np.log(np.abs(periods)), np.log(per_all), np.log(Sa_all), left=-np.nan, right=-np.nan)) 
            SigmaT = np.interp(np.log(np.abs(periods)), np.log(per_all), sigT_all,       left=-np.nan, right=-np.nan)
            SigmaS = np.interp(np.log(np.abs(periods)), np.log(per_all), sigS_all,       left=-np.nan, right=-np.nan)
    
        return (Sa, SigmaT, SigmaS)