
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.optimize import minimize

import os


class model:
	def __init__(self,model_name,datafile_name):
		'''This is general class of model for 2-afc bandit task where there is only 1 probability value to track'''
		self.model_name=model_name
		self.datafile_name=datafile_name
		self.zero=0.001 # minimum number so log(prob_choice) is not inf

	def sig(self,x):
		return(1.0/(1.0+np.exp(-1.0*x)))
	def load_data(self):
		'''
		Args:
		data - a pandas dataframe with outcomes, mag1, mag2
		'''
		self.data = pd.read_csv(self.datafile_name)

		### REMOVE DATA ####
		self.data = self.data.loc[~self.data['choice'].isnull(),:]

		print(len(self.data))
		self.outcomes = self.data['green_outcome'].as_matrix()
		self.mag_1 = self.data['green_mag'].as_matrix()
		self.mag_0 = self.data['blue_mag'].as_matrix()

		if 'block' in self.data:
			self.block=self.data['block'].as_matrix() # 1's for stable, 0 for volatile #
			self.block[self.block=='stable']=1
			self.block[self.block=='volatile']=0
		# probability of outcome 1, most, if not all, models estimate this quantity
		self.estimate_r = np.zeros(len(self.outcomes))
		if 'choice' in self.data:
			self.participant_choice = self.data['choice']

	def calc_ev(self):
		# plain ev calculation
		self.ev_diff = self.estimate_r*self.mag_1 - (1-self.estimate_r)*self.mag_0

		# ev calculation with risk preference
		if hasattr(self,'param_risk_preference'):
			self.estimate_r_scaled = np.minimum(np.maximum((self.estimate_r-0.5)*self.param_risk_preference+0.5,0),1);
			self.ev_diff = self.estimate_r_scaled*self.mag_1 - (1-self.estimate_r_scaled)*self.mag_0
		# ev calculation with 2 risk preferences
		if hasattr(self,'param_risk_preference1'):
			self.estimate_r_scaled = np.empty(len(self.estimate_r))
			self.estimate_r_scaled[self.block==1] = np.minimum(np.maximum((self.estimate_r[self.block==1]-0.5)*self.param_risk_preference1+0.5,0),1);
			self.estimate_r_scaled[self.block==0] = np.minimum(np.maximum((self.estimate_r[self.block==0]-0.5)*self.param_risk_preference0+0.5,0),1);
			self.ev_diff = self.estimate_r_scaled*self.mag_1 - (1-self.estimate_r_scaled)*self.mag_0

	def calc_prob_choice_softmax(self):
		''' calculates the probabiltiy of a choice using softmax - given ev differences'''
		# plain choice prob calculation
		if hasattr(self,'param_temp'):
			self.prob_choice= 1.0/(1+np.exp(-1*(1.0/self.param_temp)*self.ev_diff))

		# choice with 2 temperatures
		if hasattr(self,'param_temp1'):
			self.prob_choice = np.empty(len(self.estimate_r))
			self.prob_choice[self.block==1]= 1.0/(1+np.exp(-1*(1.0/self.param_temp1)*self.ev_diff[self.block==1]))
			self.prob_choice[self.block==0]= 1.0/(1+np.exp(-1*(1.0/self.param_temp0)*self.ev_diff[self.block==0]))


	def calc_loglik(self):
		''' calculates loglikhood of participants choices given probabilty of choice'''

		y = self.participant_choice
		yhat = self.prob_choice
		self.n = len(self.participant_choice)
		self.k = len(self.params_init)
		self.loglik = (y*np.log(yhat+self.zero) + (1-y)*np.log(1-yhat+self.zero)).sum()
		self.yhat_null = y.sum()/len(y)
		self.logliknull = (y*np.log(self.yhat_null) + (1-y)*np.log(1-self.yhat_null)).sum()
		self.BIC = np.log(self.n)*self.k - 2.0*self.loglik
		self.AIC = 2.0*self.k - 2.0*self.loglik
		self.pR2 = 1 - (self.loglik/self.logliknull)
		self.pred_acc = np.mean(np.round(yhat,0)==y)

	def generate_data(self):
		''' makes choices based on probability of choice '''
		self.generative_choices = np.zeros(len(self.outcomes))
		for trial in range(len(self.outcomes)):
			self.generative_choices[trial] = stats.bernoulli(self.prob_choice[trial]).rvs()

	def save_results(self,savefolder):
		filename = os.path.basename(self.datafile_name)
		savename = savefolder+self.model_name+filename
		traj_table = pd.DataFrame(data=self.estimate_r,columns=['r_est'])
		traj_table.to_csv(savename+'_trajtable.csv')
		single_parameter_table = pd.DataFrame(data=np.array([0]),columns=['test'])
		single_parameter_table['AIC'] = self.AIC
		single_parameter_table['BIC'] = self.BIC
		single_parameter_table['negLl'] = self.loglik*-1.0
		single_parameter_table.to_csv(savename+'_table.csv')

# Not part of these notes
# class model_rw(model):
# 	def run_inference(self):
# 		''' estimates r (ie. p(outcome))'''
# 		self.estimate_r = np.zeros([self.outcomes.size])
#
# 		if hasattr(self,'param_prior_r'):
# 			self.estimate_r[0] = self.param_prior_r
# 		else:
# 			self.estimate_r[0] = 0.5
#
# 		self.lr_storage = np.zeros([self.outcomes.size])
# 		self.lr_storage[0]=np.nan
# 		for ii in range(len(self.outcomes)-1):
# 			ii = ii+1
# 			if hasattr(self,'param_alpha1'):
# 			        if self.block[ii]==1:
# 			             learnrate =  self.param_alpha1
# 			        else:
# 			             learnrate =  self.param_alpha0
# 			else:
# 				learnrate = self.param_alpha
# 			self.estimate_r[ii] = self.estimate_r[ii-1] + learnrate*(self.outcomes[ii-1]-self.estimate_r[ii-1])
# 		        self.lr_storage[ii] = learnrate
#
# 	def negloglik(self,param):
# 		if self.subclass==0:
# 			self.param_temp = param[0]
# 			self.param_risk_preference=param[1]
# 			self.param_alpha = param[2]
# 		if self.subclass==1:
# 			self.param_temp1 = param[0]
# 			self.param_temp0 = param[1]
# 			self.param_risk_preference1=param[2]
# 			self.param_risk_preference0=param[3]
# 			self.param_alpha1 = param[4]
# 			self.param_alpha0 = param[5]
# 		if self.subclass==2:
# 			self.param_temp = param[0]
# 			self.param_risk_preference=param[1]
# 			self.param_alpha1 = param[2]
# 			self.param_alpha0 = param[3]
#
# 		self.run_inference()
# 		self.calc_ev()
# 		self.calc_prob_choice_softmax()
# 		self.calc_loglik()
# 		cost = -1*np.sum(self.loglik)
#
# 		return(cost)
#
# 	def specify_model_subclass(self,subclass):
# 		self.subclass=subclass
# 		if subclass==0:
# 			# single params
# 			self.params_init = [1.0,1.0,0.1]
# 			self.bnds = ((0.001,100),(0.001,10),(0.001,.999))
# 			self.params_names = ['invtemp','risk_preference','alpha']
#
# 		if subclass==1:
# 			# double params
# 			self.params_init= [1.0,1.0,1.0,1.0,0.1,0.1]
# 			self.bnds = ((0.0001,100),(0.0001,100),(0.001,10),(0.001,10),(0.001,.9999),(0.001,.9999))
# 			self.params_names = ['invtemp_stab','invtemp_vol','risk_preference_stab',
# 'risk_preference_vol','alpha_stab','alpha_vol']
#
# 		if subclass==2:
# 			# double learning rate
# 			self.params_init= [1.0,1.0,0.1,0.1]
# 			self.bnds = ((0.0001,100),(0.001,10),(0.001,.9999),(0.001,.9999))
# 			self.params_names = ['invtemp','risk_preference','alpha_stab','alpha_vol']
#
#
#
# 	def fit_parameters(self):
# 		self.res = minimize(self.negloglik, self.params_init, method='SLSQP',bounds=self.bnds)
# 		self.params_fitted = self.res.x
# 		self.negloglik(self.params_fitted) # run with fitted parameters
#
#

class model_switching(model):

	def initialize_parameter_range(self):

		# set up parameter ranges to do inference over
		self.q_range = np.arange(0,1,0.01)
		self.v_range = np.arange(0,.2,.01)

		# grid of values to evaluate them on
		self.grid_q,self.grid_v = np.meshgrid(self.q_range,self.v_range)


	def initialize_transition(self):
		# this is the transition from qt-1 to qt
		# if qt-1 is .8, then qt is either .8 or .2
		# and it is according to v #
		self.trans_func = np.zeros((self.grid_q.shape[0],self.grid_q.shape[1],self.grid_q.shape[1]))

		for vi,v in enumerate(self.v_range):
		    t1 = np.eye((self.grid_q.shape[1]))*(1-v)
		    t2 = np.rot90(np.eye((self.grid_q.shape[1]))*v)
		    self.trans_func[vi,:,:]=t1+t2

	def initialize_priors(self):

		# create prior #
		self.prior_dist = np.ones((self.grid_q.shape[0],self.grid_q.shape[1],len(self.outcomes)))

		self.prior_dist[:,:,0] = self.prior_dist[:,:,0]/(self.prior_dist.shape[0]*self.prior_dist.shape[1]) # uniform prior

		# non uniform prior
		if hasattr(self,'param_v_mu0'):

			for qi,q in enumerate(self.q_range):
				self.prior_dist[:,qi,0]	=stats.norm(loc=self.param_v_mu0,scale=self.param_v_sigma0).pdf(self.v_range)

			for vi,v in enumerate(self.v_range):
				self.prior_dist[vi,:,0]	=self.prior_dist[vi,:,0]*stats.norm(loc=self.param_q_mu0,scale=self.param_q_sigma0).pdf(self.q_range)

			self.prior_dist[:,:,0] = self.prior_dist[:,:,0]/self.prior_dist[:,:,0].sum()

		self.prior_dist[:,:,1:] = self.prior_dist[:,:,1:]*np.nan # pre-allocate where prior will be updated.

		# posteriors and marginal matrices
		self.post_dist = np.ones((self.grid_q.shape[0],self.grid_q.shape[1],len(self.outcomes)))*np.nan
		self.marg_q = np.zeros((len(self.q_range),len(self.outcomes)))
		self.marg_v = np.zeros((len(self.v_range),len(self.outcomes)))

	def run_inference(self):
		''' estimates r (ie. p(outcome))'''
		# exact inference (grid search) for posterior on each trial
		for trial,y in enumerate(self.outcomes):

			lik = self.grid_q**(y)*(1-self.grid_q)**(1-y) # bernouli distribution for likelihood

			if trial>0:
				# apply transition function from old posterior to new prior #
				old_post = self.post_dist[:,:,trial-1]
				old_post_unpacked = np.repeat(old_post[:,:,np.newaxis],len(self.q_range),axis=2)
				# dim are v,qt-1,qt, expanded old v and qt-1 into new qt

				new_prior_unpacked = np.multiply(old_post_unpacked,self.trans_func)
				# multiply old posterior by probaility that q stayed the same #
				new_prior = np.sum(new_prior_unpacked,axis=1) # sum out qt-1
				self.prior_dist[:,:,trial] = new_prior/np.sum(new_prior) # normalized again

			# mutliple prior and liklihood
			self.post_dist[:,:,trial] = np.multiply(lik,self.prior_dist[:,:,trial])
			self.post_dist[:,:,trial] = self.post_dist[:,:,trial]/np.sum(self.post_dist[:,:,trial]) # normalize

			self.marg_q[:,trial]=np.sum(self.post_dist[:,:,trial],axis=0)
			self.marg_v[:,trial]=np.sum(self.post_dist[:,:,trial],axis=1)

		self.ev_q = np.dot(self.marg_q.T,self.q_range[:,np.newaxis])
		self.ev_v = np.dot(self.marg_v.T,self.v_range[:,np.newaxis])

		# estimate for r
		self.estimate_r = self.ev_q[:,0]

	def negloglik(self,params): # change name to negloglik

		# feed in initialized model.
		if self.subclass==0:
		    self.param_temp = params[0]
		if self.subclass==1: # fitted prior
		    self.param_temp = params[0]
		    self.param_v_mu0	= params[1]
		    self.param_v_sigma0 = params[2]
		    self.param_q_mu0=0.5
		    self.param_q_sigma0=1
		    self.initialize_priors() #only have to run inference for this one again
		    self.run_inference()
		if self.subclass==2:
		    self.param_temp = params[0]
		    self.param_risk_preference=params[1]
		if self.subclass==3: # fitted prior
		    self.param_temp = params[0]
		    self.param_risk_preference=params[1]
		    self.param_v_mu0	= params[2]
		    self.param_v_sigma0 = params[3]
		    self.param_q_mu0=0.5
		    self.param_q_sigma0=1
		    self.initialize_priors() #only have to run inference for this one again
		    self.run_inference()
		self.calc_ev() # recalculate ev with new params
		self.calc_prob_choice_softmax()
		self.calc_loglik()
		cost = -1*np.sum(self.loglik)
		return(cost)


	def specify_model_subclass(self,subclass):
		self.subclass=subclass
		if subclass==0:
			# no flexible priors, just softmax
			self.params_init = [1.0]
			self.bnds = ((0,100),)
			self.params_names = ['invtemp']

		if subclass==1:
			# flexible priors and softmax,
			self.params_init = [1.0,0.1,0.02]
			self.bnds = ((0,100),(0.01,0.2),(0.001,0.3))
			self.params_names = ['invtemp','switch_rate_prior_mean','switch_rate_prior_var']

		if subclass==2:
			# flexible priors and softmax,
			self.params_init = [1.0,1.0]
			self.bnds = ((0,100),(0.001,10))
			self.params_names = ['invtemp','risk preference']

		if subclass==3:
			# flexible priors and softmax,
			self.params_init = [1.0,1.0,0.1,0.02]
			self.bnds = ((0,100),(0.001,10),(0.01,0.2),(0.001,0.3))
			self.params_names = ['invtemp','risk preference','switch_rate_prior_mean','switch_rate_prior_var']


	def fit_parameters(self):

		# fit parameters by maximize likelihood
		self.initialize_parameter_range()
		self.initialize_transition() # this takes awhile so do it here, rather than inside optimization
		self.initialize_priors() #only have to run inference for this one again
		self.run_inference()
		self.res = minimize(self.negloglik, self.params_init, method='SLSQP',bounds=self.bnds)
		self.params_fitted = self.res.x
		self.negloglik(self.params_fitted) # run with fitted parameters



class model_gaussian_rw(model):


	def initialize_parameter_range(self):

		''' set up parameter ranges to do inference over'''

		## R
		self.rmin=-7.0
		self.rmax=7.0
		rsize=30.0
		r_sig_range = np.arange(self.sig(self.rmin),self.sig(self.rmax),(self.sig(self.rmax)-self.sig(self.rmin))/rsize)
		self.r_range = np.arange(self.rmin,self.rmax,(self.rmax-self.rmin)/rsize)

		## V
		self.vmin=0.01
		self.vmax=10
		vsize=33
		self.v_range = np.arange(np.log(self.vmin),np.log(self.vmax),(np.log(self.vmax)-np.log(self.vmin))/vsize)

		## K
		self.kmin=0.0001
		self.kmax=10
		ksize=32
		self.k_range = np.arange(np.log(self.kmin),np.log(self.kmax),(np.log(self.kmax)-np.log(self.kmin))/ksize)

		## Mesh Grid
		self.grid_r = np.repeat(self.r_range[:,np.newaxis],len(self.v_range),axis=1)
		self.grid_r = np.repeat(self.grid_r[:,:,np.newaxis],len(self.k_range),axis=2)


	def initialize_transition(self):


		# P(V | V_-1, K)
		self.voltrans =np.zeros((len(self.v_range),len(self.v_range),len(self.k_range)))
		for k_i,k in enumerate(self.k_range):
			for v_1_i,v_1 in enumerate(self.v_range):
				# located given v-1, what is probability
				self.voltrans[:,v_1_i,k_i] = stats.norm(loc=v_1,scale=np.sqrt(np.exp(k))).pdf(self.v_range)
				# normalize
				self.voltrans[:,v_1_i,k_i]=self.voltrans[:,v_1_i,k_i]/self.voltrans[:,v_1_i,k_i].sum()

		# dimensions correspond to: rt,vt, rt-1,vt-1,k
		self.voltrans_5d = np.repeat(self.voltrans[np.newaxis,:,:,:],
			                              len(self.r_range),axis=0)
		self.voltrans_5d = np.repeat(self.voltrans_5d[:,:,np.newaxis,:,:],
			                              len(self.r_range),axis=2)

		## P(R | R-1, V)
		# dimensions correspond to: r_i,v_i, r_i_1
		self.rtrans =np.zeros((len(self.r_range),len(self.v_range),len(self.r_range)))
		for v_i,v in enumerate(self.v_range):
			for r_1_i,r_1 in enumerate(self.r_range):
				self.rtrans[:,v_i,r_1_i] = stats.norm(loc=r_1,scale=np.sqrt(np.exp(v))).pdf(self.r_range)
				self.rtrans[:,v_i,r_1_i]=self.rtrans[:,v_i,r_1_i]/self.rtrans[:,v_i,r_1_i].sum()

		# dimensions correspond to: r_i,v_i, r_i_1,v_i_1,k
		self.rtrans_5d = np.repeat(self.rtrans[:,:,:,np.newaxis],
			                              len(self.v_range),axis=3)

		self.rtrans_5d = np.repeat(self.rtrans_5d[:,:,:,:,np.newaxis],
			                              len(self.k_range),axis=4)

	def initialize_priors(self):

		# pre-allocate posterior matrix
		self.post_dist = np.ones((self.grid_r.shape[0],self.grid_r.shape[1],self.grid_r.shape[2],len(self.outcomes)+1))*np.nan

		# create uniform prior #
		self.prior_dist = np.ones((self.grid_r.shape[0],self.grid_r.shape[1],self.grid_r.shape[2],len(self.outcomes)+1))
		self.prior_dist[:,:,:,0] = self.prior_dist[:,:,:,0]/(self.prior_dist.shape[0]*self.prior_dist.shape[1]*self.prior_dist.shape[2])
		self.prior_dist[:,:,:,1:]= self.prior_dist[:,:,:,1:]*np.nan

		# non uniform prior
		if hasattr(self,'param_v_mu0'):
			#bias the prior on k
			for ri,r in enumerate(self.r_range):
				for vi,r in enumerate(self.v_range):
					self.prior_dist[ri,vi,:,0]=stats.norm(self.param_v_mu0,self.param_v_sigma0).pdf(self.k_range)
					self.prior_dist[ri,vi,:,0]=self.prior_dist[ri,vi,:,0]/np.sum(self.prior_dist[ri,vi,:,0])

			# bias the prior on v
			for ri,r in enumerate(self.r_range):
				for ki,k in enumerate(self.k_range):
					self.prior_dist[ri,:,ki,0]=stats.norm(self.param_k_mu0,self.param_k_sigma0).pdf(self.v_range)
					self.prior_dist[ri,:,ki,0]=self.prior_dist[ri,:,ki,0]/np.sum(self.prior_dist[ri,:,ki,0])
			# normalize over v,k biases.
			for ri,r in enumerate(self.r_range):
			    self.prior_dist[ri,:,:,0] = self.prior_dist[ri,:,:,0]/np.sum(self.prior_dist[ri,:,:,0])

		self.marg_r = np.empty((len(self.r_range),len(self.outcomes)+1))
		self.marg_v = np.empty((len(self.v_range),len(self.outcomes)+1))
		self.marg_k = np.empty((len(self.k_range),len(self.outcomes)+1))


	def run_inference(self):
		''' estimates r (ie. p(outcome))'''

		for trial,y in enumerate(self.outcomes):

			# get p(y|q) liklihood
			lik = self.sig(self.grid_r)**(y)*(1-self.sig(self.grid_r))**(1-y) # needs to be repeated across 3D for r,v,k

			if trial>0:
			# apply transition function from old posterior to new prior #
				method = 1
				if method==1:
					old_post = self.post_dist[:,:,:,trial-1]
					old_post_unpacked = np.repeat(old_post[np.newaxis,:,:,:],
					len(self.v_range),axis=0) # repeats along vt

					four_joint = np.multiply(old_post_unpacked,self.voltrans_5d[0,:,:,:,:])
					# vt, rt-1,vt-1,k

					three_joint = four_joint.sum(2)
					# vt,rt-1,k

					four_joint = np.repeat(three_joint[np.newaxis,:,:,:],len(self.r_range),axis=0) # repeats along new rt
					# rt,vt,rt-1,k

					# multiply everything together
					four_joint2 = np.multiply(four_joint,self.rtrans_5d[:,:,:,0,:]) #
					# rt, vt,rt-1,k

					# integrate out vt-1 and then rt-1 (not sure if order matters)
					three_joint2 = np.sum(four_joint2,axis=2)

					# normalize
					self.prior_dist[:,:,:,trial]=three_joint2/np.sum(three_joint2)

				elif method==2:
					# this is equivalent in a big 5D grid, much slower
					old_post = self.post_dist[:,:,:,trial-1]
					old_post_unpacked = np.repeat(old_post[np.newaxis,:,:,:],
					len(v_range),axis=0) # repeats along vt
					old_post_unpacked = np.repeat(old_post_unpacked[np.newaxis,:,:,:,:],
					len(r_range),axis=0) # repeats along vt

					five_joint = np.multiply(np.multiply(old_post_unpacked,self.voltrans_5d),self.rtrans_5d)
					three_joint = np.sum(np.sum(five_joint,axis=3),axis=2)
					self.prior_dist[:,:,:,trial]=three_joint/np.sum(three_joint)

			# mutliple prior and liklihood
			self.post_dist[:,:,:,trial] = np.multiply(lik,self.prior_dist[:,:,:,trial])
			self.post_dist[:,:,:,trial] = self.post_dist[:,:,:,trial]/np.sum(self.post_dist[:,:,:,trial]) # normalize

			self.marg_r[:,trial]=np.sum(self.post_dist[:,:,:,trial],axis=(1,2))
			self.marg_v[:,trial]=np.sum(self.post_dist[:,:,:,trial],axis=(0,2))
			self.marg_k[:,trial]=np.sum(self.post_dist[:,:,:,trial],axis=(0,1))

		self.ev_r = np.dot(self.marg_r.T,self.r_range[:,np.newaxis])
		self.ev_v= np.dot(self.marg_v.T,self.v_range[:,np.newaxis])
		self.ev_k = np.dot(self.marg_k.T,self.k_range[:,np.newaxis])
		self.estimate_r = self.sig(self.ev_r[0:-1,0]) # don't take the last trial's prediction for the next trial

	def negloglik(self,params): # change name to negloglik

		# feed in initialized model.
		if self.subclass==0:
		    self.param_temp = params[0]
		if self.subclass==2:
		    self.param_temp = params[0]
		    self.param_risk_preference=params[1]
		if self.subclass==1: # fitted prior on v, k
		    self.param_temp = params[0]
		    self.param_v_mu0	= params[1]
		    self.param_v_sigma0 = params[2]
		    #self.param_r_mu0=0.5
		    #self.param_r_sigma0=1
		    self.param_k_mu0= params[3]
		    self.param_k_sigma0 = params[4]
		    self.initialize_priors() #only have to run inference for this one again
		    self.run_inference()

		self.calc_ev() # recalculate ev with new params
		self.calc_prob_choice_softmax()
		self.calc_loglik()
		cost = -1*np.sum(self.loglik)
		return(cost)


	def specify_model_subclass(self,subclass):
		self.subclass=subclass
		if subclass==0:
			# no flexible priors, just softmax
			self.params_init = [1.0]
			self.bnds = ((0,100),)
			self.params_names = ['invtemp']

		if subclass==1:
			# flexible priors and softmax,
			self.params_init = [1.0,self.v_range[10],1,self.k_range[10],1]
			v_range_actual = self.v_range[-1]-self.v_range[0]
			k_range_actual = self.k_range[-1]-self.k_range[0]
			self.bnds = ((0,100),(self.v_range[0],self.v_range[-1]),(v_range_actual/self.v_range[-1],v_range_actual*self.v_range[-1]),(self.k_range[0],self.k_range[-1]),(k_range_actual/self.k_range[-1],k_range_actual*self.k_range[-1]))
			self.params_names = ['invtemp','volatility_prior_mean','volatility_prior_var',
					'meta_volatility_prior_mean','meta_volatility_prior_var']
		if subclass==2:
			# flexible priors and softmax,
			self.params_init = [1.0,1.0]
			self.bnds = ((0,100),(0.001,10))
			self.params_names = ['invtemp','risk preference']


	def fit_parameters(self):

		# fit parameters by maximize likelihood
		self.initialize_parameter_range()
		self.initialize_transition() # this takes awhile so do it here, rather than inside optimization
		self.initialize_priors() #only have to run inference for this one again
		self.run_inference()
		self.res = minimize(self.negloglik, self.params_init, method='SLSQP',bounds=self.bnds)
		self.params_fitted = self.res.x
		self.negloglik(self.params_fitted) # run with fitted parameters
