# Module for class-based solvers for different Inverse Ising methods.
from __future__ import division
from scipy.optimize import minimize,fmin_ncg
import scipy.optimize 
import multiprocess as mp
from utils import *
from samplers import *
import copy
import meanFieldIsing


class Solver(object):
    """
    Base class for declaring common methods and attributes.

    Params:
    -------
    n (int)
        System size.
    calc_observables (function)
        Lambda function 
        lambda params: return observables
    multipliers (ndarray=None)
    n_jobs (int=None)

    Attributes:
    -----------
    constraints (ndarray)
    calc_e (function)
        Takes states and parameters to calculate the energies.
    calc_observables (function)
        takes in n_samples as argument and returns array of (n_samples,n_constraints)
    multipliers (ndarray)
        set the Langrangian multipliers
    """
    def __init__(self, n,
                 calc_de=None,
                 calc_observables=None,
                 adj=None,
                 multipliers=None,
                 n_jobs=None):
        # Do basic checks on the inputs.
        assert type(n) is int
        
        self.n = n
        self.multipliers = multipliers
        
        self.calc_observables = calc_observables
        self.calc_e = lambda s,multipliers:-self.calc_observables(s).dot(multipliers)
        self.calc_de = calc_de
        self.adj = adj
        
        self.n_jobs = n_jobs or mp.cpu_count()

    def solve(self):
        return
              
    def estimate_jac(self,eps=1e-3):
        """
        Jacobian is an n x n matrix where each row corresponds to the behavior
        of fvec wrt to a single parameter.
        For calculation, seeing Voting I pg 83
        """
        raise NotImplementedError
        dlamda = np.zeros(self.multipliers.shape)
        jac = np.zeros((self.multipliers.size,self.multipliers.size))
        print "evaluating jac"
        for i in xrange(len(self.multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.samples,dlamda)     

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.samples,dlamda)     

            jac[i,:] = (dConstraintsPlus-dConstraintsMinus)/(2*eps)
            dlamda[i] += eps
        return jac

    def setup_sampler(self,
                      sample_method=None,
                      sampler_kwargs={},
                      optimize_kwargs={}):
        """
        Instantiate sampler class object.

        Params:
        -------
        sample_method (str)
            'wolff', 'metropolic', 'remc'
        sampler_kwargs (dict)
        optimize_kwargs (dict)
        """
        sample_method = sample_method or self.sampleMethod
        
        if sample_method=='wolff':
            raise NotImplementedError("Need to update call.")
            h,J = self._multipliers[:self.n],self.multipliers[self.n:]
            self.sampler = WolffIsing( J,h )

        elif sample_method=='metropolis':
            self.sampler = MCIsing( self.n,self.multipliers,self.calc_e )
        
        elif sample_method=='remc':
            self.sampler = ParallelTempering( self.n,
                                              self._multipliers,
                                              self.calc_e,
                                              sampler_kwargs['temps'],
                                              sample_size=self.sampleSize )
            # Parallel tempering needs to optimize choice of temperatures.
            self.sampler.optimize(**optimize_kwargs)
            
        else:
           raise NotImplementedError("Unrecognized sampler.")

    def generate_samples(self,n_iters,burnin,
                         sample_size=None,
                         sample_method=None,
                         initial_sample=None,
                         generate_kwargs={}):
        """
        Wrapper around generate_samples_parallel() from available samplers.

        Params:
        -------
        n_iters (int)
        burnin (int) 
            I think burn in is handled automatically in REMC.
        sample_size (int)
        sample_method (str)
        initial_sample (ndarray)
        generate_kwargs (dict)

        Returns:
        --------
        None
        """
        assert not (self.sampler is None), "Must call setup_sampler() first."

        sample_method = sample_method or self.sampleMethod
        sample_size = sample_size or self.sampleSize
        if initial_sample is None and (not self.samples is None) and len(self.samples)==sample_size:
            initial_sample = self.samples
        
        if sample_method=='wolff':
            self.sampler.update_parameters(self._multipliers[self.n:],self.multipliers[:self.n])
            # Burn in.
            self.samples = self.sampler.generate_sample_parallel( sample_size,burnin,
                                                                  initial_sample=initial_sample )
            self.samples = self.sampler.generate_sample_parallel( sample_size,n_iters,
                                                                  initial_sample=self.sampler.samples )

        elif sample_method=='metropolis':
            self.sampler.theta = self._multipliers
            # Burn in.
            self.sampler.generate_samples_parallel( sample_size,
                                                    n_iters=burnin,
                                                    cpucount=self.n_jobs,
                                                    initial_sample=initial_sample )
            self.sampler.generate_samples_parallel( sample_size,
                                                    n_iters=n_iters,
                                                    cpucount=self.n_jobs,
                                                    initial_sample=self.sampler.samples)
            self.samples = self.sampler.samples

        elif sample_method=='remc':
            self.sampler.update_parameters(self._multipliers)
            self.sampler.generate_samples(n_iters=n_iters,**generate_kwargs)
            self.samples = self.sampler.replicas[0].samples

        else:
           raise NotImplementedError("Unrecognized sampler.")
# end Solver



class Exact(Solver):
    """
    Class for solving +/-1 symmetric Ising model maxent problems by gradient descent with flexibility to put
    in arbitrary constraints.

    Params:
    -------
    n (int)
        System size.
    constraints (ndarray)
    calc_e (function)
        lambda samples,params: return energy
    calc_observables (function)
        For exact: lambda params: return observables

    Attributes:
    -----------
    constraints (ndarray)
    calc_e (function)
        with args (sample,parameters) where sample is 2d
    calc_observables (function)
        takes in samples as argument
    multipliers (ndarray)
        set the Langrangian multipliers
    """
    def __init__(self, *args, **kwargs):
        self.calc_observables_multipliers = kwargs['calc_observables_multipliers']
        del kwargs['calc_observables_multipliers']
        super(Exact,self).__init__(*args,**kwargs)

    def solve(self,
              constraints=None,
              samples=None,
              initial_guess=None,
              tol=None,
              tolNorm=None,
              disp=False,
              max_param_value=50,
              fsolve_kwargs={'method':'powell'}):
        """
        Params:
        ------
        constraints (array-like)
        samples (array-like)
            (n_samples,n_dim)
        initial_guess (ndarray=None)
            initial starting point
        tol (float=None)
            maximum error allowed in any observable
        tolNorm (float)
            norm error allowed in found solution
        nIters (int=30)
            number of iterations to make when sampling
        disp (bool=False)
        fsolve_kwargs (dict={'method':'powell'})
            Powell method is slower but tends to converge better.

        Returns:
        --------
        Tuple of solved parameters and output from scipy.optimize.minimize
        """
        if not constraints is None:
            self.constraints = constraints
        elif not samples is None:
            self.constraints = self.calc_observables(samples).mean(0)
        else:
            raise Exception("Must specify either constraints or samples.")
        
        if not initial_guess is None:
            self.multipliers = initial_guess.copy()
        else: initial_guess = np.zeros((len(self.constraints)))
        
        def f(params):
            if np.any(np.abs(params)>max_param_value):
                return [1e30]*len(params)
            return np.linalg.norm( self.calc_observables_multipliers(params)-self.constraints )

        soln = minimize(f,initial_guess,**fsolve_kwargs)
        return soln['x'],soln
# End Exact



def unwrap_self_worker_obj(arg, **kwarg):
    return MPF.worker_objective_task(*arg, **kwarg)

class MPF(Solver):
    def __init__(self, *args, **kwargs):
        """
        Parallelized implementation of Minimum Probability Flow algorithm.
        Slowest step is the computation of the energy of a given state. Make this as fast as possible.

        Params:
        -------
        calc_e (lambda state,params)
            function for computing energies of given state and parameters.  Should take in a 2D state array
            and vector of parameters to compute energies.
        adj (lambda state)
            function for getting all the neighbors of any given state
        calc_de (lambda=None)
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        n_jobs (int=0)
            If 0 no parallel processing, other numbers above 0 specify number of cores to use.
        
        Attributes:
        -----------
        
        Methods:
        --------
        """
        super(MPF,self).__init__(*args,**kwargs)
        
    @staticmethod
    def worker_objective_task( s, Xcount, adjacentStates, params, calc_e ):
        return Xcount * np.sum(np.exp( .5*(calc_e(s[None,:],params) 
                                           - calc_e(adjacentStates,params) ) ))
 
    def K( self, Xuniq, Xcount, adjacentStates, params ):
        """
        Compute objective function.
        
        Params:
        -------
        Xuniq (ndata x ndims ndarray)
            unique states that appear in the data
        Xcount (ndarray of ints)
            number of times that each unique state appears in the data
        adjacentStates (list of ndarrays)
            list of adjacent states for each given unique state
        params (ndarray)
            parameters for computation of energy
        """
        if self.pool is None:
            obj = 0.
            objGrad = np.zeros((params.size))
            for i,s in enumerate(Xuniq):
                dobj = Xcount[i] * np.exp( .5*(self.calc_e(s[None,:],params) 
                                               - self.calc_e(adjacentStates[i],params) ) )
                if not self.calc_de is None:
                    for j in xrange(params.size):
                        if dobj.size!=adjacentStates[i].shape[0]:
                            raise Exception("Sizes do not match")
                        objGrad[j] += .5 * (dobj * ( self.calc_de(s[None,:],j) 
                                            - self.calc_de(adjacentStates[i],j) )).sum()
                obj += dobj.sum()
        else:
            # Parallel loop through objective function calculation for each state in the data.
            obj = [self.pool.apply( unwrap_self_worker_obj, 
                                    args=([Xuniq[i],Xcount[i],adjacentStates[i],params,self.calc_e],) ) 
                        for i in xrange(Xuniq.shape[0])]
            obj = obj.sum()

            if not self.calc_de is None:
                from warning import warn
                warn("Gradient computation not written fro parallel loop.")

        if not self.calc_de is None:
            return obj / Xcount.sum(), objGrad / Xcount.sum()
        else:
            return obj / Xcount.sum()
       
    def _K( self, X, J ):
        """
        Translation from Sohl-Dickstein's code K_dk_ising.m. This is here for testing purposes only.
        Caution: This uses a different convention for negatives and 1/2 factors. To use this properly, all
        parameters will have an extra negative, the returned J's will be halved and the energy calculation
        should include a 1/2 factor in front of h's.
        """
        nbatch, ndims = X.shape
        X = X.T
        
        h = J[:ndims]
        J = squareform( J[ndims:] )
        J[diag_indices(ndims)] = h
        
        Y = dot(J,X)
        diagJ = J.diagonal()
    #     % XnotX contains (X - [bit flipped X])
        XnotX = 2.*X-1;
    #     % Kfull is a [ndims, nbatch] matrix containing the contribution to the 
    #     % objective function from flipping each bit in the rows, for each datapoint 
    #     % on the columns
        Kfull = np.exp( XnotX * Y - (1/2)*tile(diagJ[:,None],(1,nbatch)) )
        K = sum(Kfull)
        K  = K  / nbatch
        return K

    def logK( self, Xuniq, Xcount, adjacentStates, params ):
        """
        Compute log of objective function.
        
        Params:
        -------
        Xuniq (ndata x ndims ndarray)
            unique states that appear in the data
        Xcount (ndarray of ints)
            number of times that each unique state appears in the data
        adjacentStates (list of ndarrays)
            list of adjacent states for each given unique state
        params (ndarray)
            parameters for computation of energy

        Returns:
        --------
        logK (float)
        """
        from scipy.misc import logsumexp

        obj = 0.
        objGrad = np.zeros((params.size))
        power=np.zeros((len(Xuniq),len(adjacentStates[0])))  # energy differences
        for i,s in enumerate(Xuniq):
            power[i,:] = .5*( self.calc_e(s[None,:],params) - self.calc_e(adjacentStates[i],params) )
            
        obj=logsumexp( power+np.log(Xcount)[:,None] )
        
        if not self.calc_de is None:
            # coefficients that come out from taking derivative of exp
            for i in xrange(params.size):
                gradcoef=np.zeros((len(Xuniq),len(adjacentStates[0])))  
                for j,s in enumerate(Xuniq): 
                    gradcoef[j,:] = .5 * ( self.calc_de(s[None,:],i) 
                                           - self.calc_de(adjacentStates[j],i) )
                power -= power.max()
                objGrad[i]=(gradcoef*np.exp(power)*Xcount[:,None]).sum()/(np.exp(power)*Xcount[:,None]).sum()

        if not self.calc_de is None:
            if objGrad.size==1:
                raise Exception("")
            return obj / Xcount.sum(), objGrad / Xcount.sum()
        else:
            return obj / Xcount.sum()
    # End logK

    def solve( self,
               X=None, 
               initial_guess=None,
               method='L-BFGS-B',
               all_connected=True,
               parameter_limits=100,
               solver_kwargs={'maxiter':100,'disp':True,'ftol':1e-15},
               uselog=True,
               ):
        """
        Minimize MPF objective function using scipy.optimize.minimize.

        Params:
        -------
        X (ndata x ndim ndarray)
            array of states compatible with given energy and adjacent neighbors functions
        adj (lambda state)
            returns adjacent states for any given state
        all_connected (bool=True)
            switch for summing over all states that data sets could be connected to or just summing over
            non-data states (second summation in Eq 10 in Sohl-Dickstein 2011)
        iterate (int=0)
            number of times to try new initial conditions if first try doesn't work. Right now, this is a
            pretty coarse test because the fit can be good even without converging.
        parameter_limits (float)
            some limit to constrain the space that the solver has to search. This is the maximum allowed
            magnitude of any single parameter.
        solver_kwargs (dict)
            For scipy.optimize.minimize.

        Returns:
        --------
        soln (ndarray)
            found solution to problem
        output (dict)
            full output from minimize solver
        """
        assert parameter_limits>0
        assert not X is None, "samples must be provided by MPF"

        # Convert from {0,1} to {+/-1} asis.
        X = (X+1)/2
        
        if not self.calc_de is None:
            includeGrad = True
        else:
            includeGrad = False
        X = X.astype(float)
        if initial_guess is None:
            initial_guess = self.calc_observables(X).mean(0)#np.zeros(self.n+self.n*(self.n-1)//2)
         
        # Get list of unique data states and how frequently they appear.
        Xuniq = X[unique_rows(X)]
        ix = unique_rows(X,return_inverse=True)
        Xcount = np.bincount(ix)
        M,N = Xuniq.shape
        
        adjacentStates = []
        for s in Xuniq:
            adjacentStates.append( self.adj(s) )
            # Remove states already in data.
            if not all_connected:
                ix = np.zeros((s.size))==0
                for i,t in enumerate(adjacentStates[-1]):
                    if np.any(np.all(t[None,:]==Xuniq,1)):
                        ix[i] = False
                if np.sum(ix)==X.shape[1]:
                    raise Exception("This data set does not satisfy MPF assumption that each \
                                    state be connected to at least one non-data state (?)")
                adjacentStates[-1] = adjacentStates[-1][ix]

        # Interface to objective function.
        if uselog:
            def f(params):
                return self.logK( Xuniq, Xcount, adjacentStates, params )
        else:
            def f(params):
                return self.K( Xuniq, Xcount, adjacentStates, params )
        
        # If calc_de has been provided then minimize will use gradient information.
        soln = minimize( f, initial_guess,
                         bounds=[(-parameter_limits,parameter_limits)]*len(initial_guess),
                         method=method, jac=includeGrad, options=solver_kwargs )
        # NOTE: Returning soln details in terms of {0,1} basis.
        return convert_params(soln['x'][:self.n],soln['x'][self.n:],'11',True), soln
# End MPFSolver



class MCH(Solver):
    """
    Class for solving maxent problems using the Monte Carlo Histogram method.

    Broderick, T., Dudik, M., Tkacik, G., Schapire, R. E. & Bialek, W. Faster solutions of the inverse
    pairwise Ising problem. arXiv 1-8 (2007).
    """
    def __init__(self, *args, **kwargs):
        """
        Params:
        -------
        calc_e (lambda state,params)
            function for computing energies of given state and parameters.  Should take in a 2D state array
            and vector of parameters to compute energies.
        adj (lambda state)
            function for getting all the neighbors of any given state
        calc_de (lambda=None)
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        n_jobs (int=0)
            If 0 no parallel processing, other numbers above 0 specify number of cores to use.
        
        Attributes:
        -----------
        constraints (ndarray)
        calc_e (function)
            with args (sample,parameters) where sample is 2d
        calc_observables (function)
            takes in samples as argument
        mch_approximation (function)
        sampleSize (int)
        multipliers (ndarray)
            set the Langrangian multipliers

        Methods:
        --------
        """
        sample_size,sample_method,mch_approximation = (kwargs.get('sample_size',None),
                                                       kwargs.get('sample_method',None),
                                                       kwargs.get('mch_approximation',None))
        assert not sample_size is None, "Must specify sample_size."
        assert not sample_method is None, "Must specify sample_method."
        assert not mch_approximation is None, "Must specify mch_approximation."
        del kwargs['sample_size'],kwargs['sample_method'],kwargs['mch_approximation']
        super(MCH,self).__init__(*args,**kwargs)
        assert not self.calc_observables is None, "Must specify calc_observables."
        
        self.mch_approximation = mch_approximation
        
        # Sampling parameters.
        self.sampleSize = sample_size
        self.sampleMethod = sample_method
        self.sampler = None
        self.samples = None
        
        self.setup_sampler(self.sampleMethod)
    
    def solve(self,
              constraints=None,
              X=None,
              initial_guess=None,
              tol=None,
              tolNorm=None,
              n_iters=30,
              burnin=30,
              maxiter=10,
              disp=False,
              full_output=False,
              learn_params_kwargs={},
              generate_kwargs={}):
        """
        Solve for parameters using MCH routine.
        
        NOTE: Commented part relies on stochastic gradient descent but doesn't seem to
        be very good at converging to the right answer with some tests on small systems.
        
        Params:
        ------
        initial_guess (ndarray=None)
            initial starting point
        tol (float=None)
            maximum error allowed in any observable
        tolNorm (float)
            norm error allowed in found solution
        n_iters (int=30)
            Number of iterations to make between samples in MCMC sampling.
        burnin (int=30)
        disp (bool=False)
        learn_parameters_kwargs
        generate_kwargs

        Returns:
        --------
        parameters (ndarray)
            Found solution.
        errflag (int)
        errors (ndarray)
            Errors in matching constraints at each step of iteration.
        """
        # Read in constraints.
        if not constraints is None:
            self.constraints = constraints
        elif not X is None:
            self.constraints = self.calc_observables(X).mean(0)
        
        # Set initial guess for parameters.
        if not (initial_guess is None):
            assert len(initial_guess)==len(self.constraints)
            self._multipliers = initial_guess.copy()
        else:
            self._multipliers = np.zeros((len(self.constraints)))
        tol = tol or 1/np.sqrt(self.sampleSize)
        tolNorm = tolNorm or np.sqrt( 1/self.sampleSize )*len(self._multipliers)

        errors = []  # history of errors to track
        
        self.generate_samples(n_iters,burnin,
                              generate_kwargs=generate_kwargs)
        thisConstraints = self.calc_observables(self.samples).mean(0)
        errors.append( thisConstraints-self.constraints )
        if disp=='detailed': print self._multipliers
        
        # MCH iterations.
        counter = 0
        keepLoop = True
        while keepLoop:
            if disp:
                print "Iterating parameters with MCH..."
            self.learn_parameters_mch(thisConstraints,**learn_params_kwargs)
            if disp=='detailed':
                print "After MCH step, the parameters are..."
                print self._multipliers
            if disp:
                print "Sampling..."
            self.generate_samples( n_iters,burnin,
                                   generate_kwargs=generate_kwargs )
            thisConstraints = self.calc_observables(self.samples).mean(0)
            counter += 1
            
            # Exit criteria.
            errors.append( thisConstraints-self.constraints )
            if ( np.linalg.norm(errors[-1])<tolNorm
                 and np.all(np.abs(thisConstraints-self.constraints)<tol) ):
                print "Solved."
                errflag=0
                keepLoop=False
            elif counter>maxiter:
                print "Over maxiter"
                errflag=1
                keepLoop=False
        
        if full_output:
            return self._multipliers,errflag,np.vstack((errors))
        return self._multipliers

        #def f(lamda):
        #    if np.any(np.abs(lamda)>10):
        #        return [1e30]*len(lamda)
        #    self.generate_samples(nIters=20)
        #    print "generating samples for"
        #    print lamda
        #    thisConstraints = self.calc_observables(self.samples)
        #    return thisConstraints-self.constraints

        #if initial_guess is None:
        #    initial_guess = self.multipliers
        #soln = opt.leastsq(f, initial_guess, Dfun=lambda x: self.estimate_jac(), full_output=True,**kwargs)
        #self.multipliers = soln[0]
        #return soln

    def estimate_jac(self,eps=1e-3):
        """
        Jacobian is an n x n matrix where each row corresponds to the behavior
        of fvec wrt to a single parameter.
        For calculation, seeing Voting I pg 83
        2015-08-14
        """
        dlamda = np.zeros(self._multipliers.shape)
        jac = np.zeros((self._multipliers.size,self._multipliers.size))
        print "evaluating jac"
        for i in xrange(len(self._multipliers)):
            dlamda[i] += eps
            dConstraintsPlus = self.mch_approximation(self.samples,dlamda)     

            dlamda[i] -= 2*eps
            dConstraintsMinus = self.mch_approximation(self.samples,dlamda)     

            jac[i,:] = (dConstraintsPlus-dConstraintsMinus)/(2*eps)
            dlamda[i] += eps
        return jac

    def learn_parameters_mch(self, estConstraints,
                             maxdlamda=1,
                             maxdlamdaNorm=1, 
                             maxLearningSteps=50,
                             eta=1 ):
        """
        Params:
        -------
        estConstraints (ndarray)
        maxdlamda (float=1)
        maxdlamdaNorm (float=1)
        maxLearningSteps (int)
            max learning steps before ending MCH
        eta (float=1)
            factor for changing dlamda

        Returns:
        --------
        estimatedConstraints (ndarray)
        """
        keepLearning = True
        dlamda = np.zeros((self.constraints.size))
        learningSteps = 0
        distance = 1
        
        while keepLearning:
            # Get change in parameters.
            # If observable is too large, then corresponding energy term has to go down 
            # (think of double negative).
            dlamda += -(estConstraints-self.constraints) * np.min([distance,1.]) * eta
            #dMultipliers /= dMultipliers.max()
            
            # Predict distribution with new parameters.
            estConstraints = self.mch_approximation( self.samples, dlamda )
            distance = np.linalg.norm( estConstraints-self.constraints )
                        
            # Counter.
            learningSteps += 1

            # Evaluate exit criteria.
            if np.linalg.norm(dlamda)>maxdlamdaNorm or np.any(np.abs(dlamda)>maxdlamda):
                keepLearning = False
            elif learningSteps>maxLearningSteps:
                keepLearning = False

        self._multipliers += dlamda
        return estConstraints
# End GeneralMaxentSolver



class Pseudo(Solver):
    def __init__(self, *args, **kwargs):
        """
        Pseudolikelihood approximation to solving the inverse Ising problem as described in
        Aurell and Ekeberg, PRL 108, 090201 (2012).
        
        Params:
        -------
        
        Attributes:
        -----------
        
        Methods:
        --------
        """
        super(Pseudo,self).__init__(*args,**kwargs)

    def solve(self,X=None):
        """
        Params:
        -------
        X (ndarray)
            Data set. (n_samples, n_dim)
        """
        X = (X + 1)/2  # change from {-1,1} to {0,1}
        
        # start at freq. model params?
        freqs = np.mean(X,axis=0)
        hList = -np.log(freqs/(1.-freqs))
        Jfinal = np.zeros((self.n,self.n))

        for r in xrange(self.n):
            print "Minimizing for r =",r
            
            Jr0 = np.zeros(self.n)
            Jr0[r] = hList[r]
            
            XRhat = X.copy()
            XRhat[:,r] = np.ones(len(X))
            # calculate once and pass to hessian algorithm for speed
            pairCoocRhat = self.pair_cooc_mat(XRhat)
            
            Lr = lambda Jr: - self.cond_log_likelihood(r,X,Jr)
            fprime = lambda Jr: self.cond_jac(r,X,Jr)
            fhess = lambda Jr: self.cond_hess(r,X,Jr,pairCoocRhat=pairCoocRhat)
            
            Jr = fmin_ncg(Lr,Jr0,fprime,fhess=fhess)
            Jfinal[r] = Jr

        Jfinal = -0.5*( Jfinal + Jfinal.T )
        hfinal = Jfinal[np.diag_indices(self.n)]

        # Convert parameters into {-1,1} basis as is standard.
        Jfinal[np.diag_indices(self.n)] = 0
        self.multipliers = convert_params( hfinal,squareform(Jfinal)*2,'11',concat=True )

        return self.multipliers

    def cond_log_likelihood(self,r,X,Jr):
        """
        Equals the conditional log likelihood -L_r.
        
        Params:
        -------
        r (int)
            individual index
        X (ndarray)
            binary matrix, (# X) x (dimension of system)
        Jr (ndarray)
            (dimension of system) x (1)
        """
        X,Jr = np.array(X),np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        invPs = 1. + np.exp( energies )
        logLs = np.log( invPs )

        return -logLs.sum()

    def cond_jac(self,r,X,Jr):
        """
        Returns d cond_log_likelihood / d Jr,
        with shape (dimension of system)
        """
        X,Jr = np.array(X),np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        coocs = np.repeat([sigmaRtilde],self.n,axis=0).T * samplesRhat # (#X)x(self.n)

        return np.dot( coocs.T, 1./(1. + np.exp(-energies)) )

    def cond_hess(self,r,X,Jr,pairCoocRhat=None):
        """
        Returns d^2 cond_log_likelihood / d Jri d Jrj, with shape
        (dimension of system)x(dimension of system)

        Current implementation uses more memory for speed.
        For large sample size, it may make sense to break up differently
        if too much memory is being used.

        Params:
        -------
        pairCooc (None)
            Pass pair_cooc_mat(X) to speed calculation.
        """
        X,Jr = np.array(X),np.array(Jr)
        
        sigmaRtilde = (2.*X[:,r] - 1.)
        samplesRhat = 2.*X.copy()
        samplesRhat[:,r] = np.ones(len(X))
        localFields = np.dot(Jr,samplesRhat.T) # (# X)x(1)
        energies = sigmaRtilde * localFields # (# X)x(1)
        
        # pairCooc has shape (# X)x(n)x(n)
        if pairCoocRhat is None:
            pairCoocRhat = self.pair_cooc_mat(samplesRhat)
        
        energyMults = np.exp(-energies)/( (1.+np.exp(-energies))**2 ) # (# X)x(1)
        #filteredSigmaRtildeSq = filterVec * (2.*X[:,r] + 1.) # (# X)x(1)
        return np.dot( energyMults, pairCoocRhat )

    def pair_cooc_mat(self,X):
        """
        Returns matrix of shape (self.n)x(# X)x(self.n).
        
        For use with cond_hess.
        
        Slow because I haven't thought of a better way of doing it yet.
        """
        p = [ np.outer(f,f) for f in X ]
        return np.transpose(p,(1,0,2))

    def pseudo_log_likelhood(self,X,J):
        """
        (Could probably be made more efficient.)

        Params:
        -------
        X
            binary matrix, (# of samples) x (dimension of system)
        J
            (dimension of system) x (dimension of system)
            J should be symmetric
        """
        return np.sum([ cond_log_likelihood(r,X,J) \
                           for r in xrange(len(J)) ])
# End Pseudo



class ClusterExpansion(Solver):
    def __init__(self, *args, **kwargs):
        """
        Implementation of Adaptive Cluster Expansion for
        solving the inverse Ising problem, as described in
        John Barton and Simona Cocco, J. of Stat. Mech.
        P03002 (2013).
        
        Specific to pairwise Ising constraints
        
        Params:
        -------
        calc_e (lambda state,params)
            function for computing energies of given state and parameters.  Should take in a 2D state array
            and vector of parameters to compute energies.
        adj (lambda state)
            function for getting all the neighbors of any given state
        calc_de (lambda=None)
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        n_jobs (int=0)
            If 0 no parallel processing, other numbers above 0 specify number of cores to use.
        
        Attributes:
        -----------
        
        Methods:
        --------
        """
        super(ClusterExpansion,self).__init__(*args,**kwargs)
        self.setup_sampler('metropolis')
    
    def S(self,cluster,coocMat,deltaJdict={},
          useAnalyticResults=False,priorLmbda=0.,
          numSamples=None):
        """
        Calculate pairwise entropy of cluster.
        (First fits pairwise Ising model.)
        
        useAnalyticResults (False)  : probably want False until 
                                      analytic formulas are
                                      changed to include
                                      prior on J
        """
        if len(cluster) == 0:
            raise Exception
        elif (len(cluster) == 1) and useAnalyticResults:
            p = coocMat[cluster[0],cluster[0]]
            J = np.array( [ [ -log( p / (1.-p) ) ] ] )
        elif (len(cluster) == 2) and useAnalyticResults:
            i = min(cluster[0],cluster[1])
            j = max(cluster[0],cluster[1])
            pi = coocMat[i,i]
            pj = coocMat[j,j]
            pij = coocMat[i,j]
            Jii1 = -log( pi / (1.-pi) )
            Jjj1 = -log( pj / (1.-pj) )
            Jii = -log( (pi - pij)/(1.-pi-pj+pij) )
            Jjj = -log( (pj - pij)/(1.-pi-pj+pij) )
            Jij = - log( pij ) + log( pi - pij ) + log( pj - pij )    \
                - log( 1.-pi-pj+pij )
            J = np.array( [ [ Jii, 0.5*Jij ], [ 0.5*Jij, Jjj ] ] )
        else:
            coocMatCluster = meanFieldIsing.coocCluster(coocMat,cluster)
            Jinit = None # <--- potential for speed-up here
            J = meanFieldIsing.findJmatrixAnalytic_CoocMat(coocMatCluster,
                                            Jinit=Jinit,
                                            priorLmbda=priorLmbda,
                                            numSamples=numSamples)

        # make 'full' version of J (of size NxN)
        N = len(coocMat)
        Jfull = meanFieldIsing.JfullFromCluster(J,cluster,N)
        
        ent = meanFieldIsing.analyticEntropy(J)

        return ent,Jfull 




    # 3.24.2014
    def Sindependent(self,cluster,coocMat):
        """
        """
        coocMatCluster = meanFieldIsing.coocCluster(coocMat,cluster)
        # in case we're given an upper-triangular coocMat:
        coocMatCluster = meanFieldIsing.symmetrizeUsingUpper(coocMatCluster)
        
        N = len(cluster)
        
        freqs = np.diag(coocMatCluster).copy()

        h = - np.log(freqs/(1.-freqs))
        Jind = np.diag(h)

        Sinds = -freqs*np.log(freqs)             \
            -(1.-freqs)*np.log(1.-freqs)
        Sind = np.sum(Sinds)

        # make 'full' version of J (of size NfullxNfull)
        Nfull = len(coocMat)
        Jfull = meanFieldIsing.JfullFromCluster(Jind,cluster,Nfull)

        return Sind,Jfull



    # "Algorithm 1"
    def deltaS(self,cluster,coocMat,deltaSdict=None,deltaJdict=None,
        verbose=True,meanFieldRef=False,priorLmbda=0.,
        numSamples=None,independentRef=False,meanFieldPriorLmbda=None):
        """
        cluster         : List of indices in cluster
        independentRef  : If True, expand about independent entropy
        meanFieldRef    : If True, expand about mean field entropy
        """
        if deltaSdict is None: deltaSdict = {}
        if deltaJdict is None: deltaJdict = {}
        
        if (independentRef and meanFieldRef) or \
           not (independentRef or meanFieldRef): raise Exception
        
        if meanFieldPriorLmbda is None:
            meanFieldPriorLmbda = priorLmbda
        
        cID = self.clusterID(cluster)
        if cID in deltaSdict:
            #print "deltaS: found answer for",cluster
            return deltaSdict[cID],deltaJdict[cID]
        elif verbose:
            print "deltaS: Calculating entropy for cluster",cluster
        
        # start with full entropy (and J)
        deltaScluster,deltaJcluster = self.S(cluster,coocMat,
                                        deltaJdict,
                                        priorLmbda=priorLmbda,
                                        numSamples=numSamples)
        
        if independentRef:
            # subtract independent reference entropy
            S0cluster,J0cluster = self.Sindependent(cluster,coocMat)
            deltaScluster -= S0cluster
            deltaJcluster -= J0cluster
        elif meanFieldRef:
            # subtract mean field reference entropy
            S0cluster,J0cluster = SmeanField(cluster,coocMat,
                meanFieldPriorLmbda,numSamples)
            deltaScluster -= S0cluster
            deltaJcluster -= J0cluster
        
        # subtract entropies of sub-clusters
        for size in range(len(cluster)-1,0,-1):
          subclusters = self.subsets(cluster,size)
          for subcluster in subclusters:
            deltaSsubcluster,deltaJsubcluster = \
                self.deltaS(subcluster,coocMat,deltaSdict,deltaJdict,
                       verbose=verbose,
                       meanFieldRef=meanFieldRef,priorLmbda=priorLmbda,
                       numSamples=numSamples,
                       independentRef=independentRef,
                       meanFieldPriorLmbda=meanFieldPriorLmbda)
            deltaScluster -= deltaSsubcluster
            deltaJcluster -= deltaJsubcluster

        deltaSdict[cID] = deltaScluster
        deltaJdict[cID] = deltaJcluster

        return deltaScluster,deltaJcluster

    def clusterID(self,cluster):
        return tuple(np.sort(cluster))

    def subsets(self,set,size,sort=False):
        """
        Given a list, returns a list of all unique subsets
        of that list with given size.
        """
        if len(set) != len(np.unique(set)): raise Exception
        
        if size == len(set): return [set]
        if size > len(set): return []
        if size <= 0: return []
        if size == 1: return [ [s,] for s in set ]
        
        sub = []
        rest = copy.copy(set)
        s = rest[0]
        rest.remove(s)
        
        subrest1 = self.subsets(rest,size)
        sub.extend(subrest1)
        
        subrest2 = self.subsets(rest,size-1)
        [ srest.append(s) for srest in subrest2 ]
        sub.extend(subrest2)
        
        if sort:
            return np.sort(sub)
        return sub


    # "Algorithm 2"
    # was "adaptiveClusterExpansion"
    def solve(self,X,threshold,
              cluster=None,deltaSdict=None,deltaJdict=None,
              verbose=True,priorLmbda=0.,numSamples=None,
              meanFieldRef=False,independentRef=True,veryVerbose=False,
              meanFieldPriorLmbda=None,return_all=False):
        """
        Params:
        -------
        X (array-like)
            Data set (n_samples,n_dim).
        threshold (float)
        meanFieldRef (False)
            Expand about mean-field reference
        independentRef (True)
            Expand about independent reference
        priorLmbda (0.)
            Strength of non-interacting prior
        meanFieldPriorLmbda (None)
            Strength of non-interacting prior in mean field calculation
            (defaults to priorLmbda)
        
        Returns:
        --------
        With return_all=False, returns
            J           : Estimated interaction matrix
        
        With return_all=True, returns
            ent         : Estimated entropy
            J           : Estimated interaction matrix
            clusters    : List of clusters
            deltaSdict  : 
            deltaJdict  :
        """
        # 7.18.2017 convert input to coocMat
        coocMat = self.coocurrence_matrix((X+1)/2)
        
        if deltaSdict is None: deltaSdict = {}
        if deltaJdict is None: deltaJdict = {}
        
        if independentRef and meanFieldRef: raise Exception
        
        if meanFieldPriorLmbda is None:
            meanFieldPriorLmbda = priorLmbda
        
        N = len(coocMat)
        T = threshold
        if cluster is None: cluster = range(N)

        clusters = {} # LIST
        size = 1
        clusters[1] = [ [i] for i in cluster ]

        while len(clusters[size]) > 0:
            clusters[ size+1 ] = []
            numClusters = len(clusters[size])
            if verbose:
                print "adaptiveClusterExpansion: Clusters of size", \
                    size+1
            for i in range(numClusters):
              for j in range(i+1,numClusters): # some are not unique!
                gamma1 = clusters[size][i]
                gamma2 = clusters[size][j]
                gammaI = np.intersect1d(gamma1,gamma2)
                gammaU = np.sort( np.union1d(gamma1,gamma2) )
                gammaU = list(gammaU)
                if (len(gammaI) == size-1):
                  deltaSgammaU,deltaJgammaU =                       \
                    self.deltaS(gammaU,coocMat,deltaSdict,deltaJdict,
                    verbose=veryVerbose,
                    meanFieldRef=meanFieldRef,
                    priorLmbda=priorLmbda,
                    numSamples=numSamples,
                    independentRef=independentRef,
                    meanFieldPriorLmbda=meanFieldPriorLmbda)
                  if (abs(deltaSgammaU) > T)                        \
                    and (gammaU not in clusters[size+1]):
                    clusters[ size+1 ].append(gammaU)
            size += 1
        
        if independentRef:
            ent,J0 = self.Sindependent(cluster,coocMat)
        elif meanFieldRef:
            ent,J0 = SmeanField(cluster,coocMat,
                                meanFieldPriorLmbda,numSamples)
        else:
            ent = 0.
            J0 = np.zeros((N,N))
        J = J0.copy()

        for size in clusters.keys():
            for cluster in clusters[size]:
                cID = self.clusterID(cluster)
                ent += deltaSdict[cID]
                J += deltaJdict[cID]

        # 7.18.2017 convert J to {-1,1}
        h = -J.diagonal()
        J = -meanFieldIsing.zeroDiag(J)
        self.multipliers = convert_params( h,squareform(J)*2,'11',concat=True )

        if return_all:
            return ent,self.multipliers,clusters,deltaSdict,deltaJdict
        else:
            return self.multipliers


    # 8.13.2014 took code from runSelectiveClusterExpansion
    def iterate_cluster_expansion(coocMat,
        retall=False,
        epsThreshold=1.,
        gammaPrime=0.1,
        logThresholdRange=(-6,-2),
        numThresholds=1000,
        verbose=True,veryVerbose=False,numSamplesData=None,
        numSamplesError=1e4,
        saveSamplesAtEveryStep=False,
        numProcs=1,
        minimizeIndependent=True,
        minimizeCovariance=False,
        maxMaxClusterSize=None,
        fileStr=None,
        meanFieldGammaPrime=None,
        minThreshold=0.,
        bruteForceMin=False,
        **kwargs):
      """
      ***  As of 7.18.2017, not yet converted to be usable in coniii. ***
      
      Solve adaptive cluster expansion over a range of thresholds.
      
      gammaPrime (0.1)          : Strength of noninteracting prior
      numSamplesData (None)     : Number of data samples (used to
                                  scale strength of prior)
      numSamplesError (1e4)     : Number of Ising samples used to
                                  estimate error
      bruteForceMin (False)     : If True, use J from the cluster
                                  expansion as the starting parameters
                                  for a brute force estimation
                                  (not used recently?)
      """

      thresholds = np.logspace(logThresholdRange[0],
                                  logThresholdRange[1],
                                  numThresholds)[::-1]

      thresholdIndex = 0
      stop = False
        
      deltaSdict = {}
      deltaJdict = {}
      samplesDict = {}
      epsValsList = []
      thresholdList = []
      meanFightSizeList = []
      numClustersList = []
      maxClusterSizeList = []
        
      clusters = {}
        
      bestEps = np.inf
      
      if maxMaxClusterSize is None: maxMaxClusterSize = len(coocMat)
      if meanFieldGammaPrime is None: meanFieldGammaPrime = gammaPrime # 3.25.2014
        
      # 3.31.2014 calculate prior strength
      pmean = np.mean(np.diag(coocMat))
      priorLmbda = gammaPrime / (pmean**2 * (1.-pmean)**2) #10.
      meanFieldPriorLmbda = meanFieldGammaPrime / (pmean**2 * (1.-pmean)**2) #10.

      while not stop:
        
        thresholdIndex += 1
        threshold = thresholds[thresholdIndex]
        
        if veryVerbose:
            print 'threshold =',threshold
            
        clustersOldLength = np.sum([ len(clusterlist) for clusterlist in clusters.values() ])
        
        # do fitting for decreasing thresholds
        raise Exception,"Not implemented: Need to change function call to use coniii interface"
        ent,J,clusters,deltaSdict,deltaJdict = \
            self.solve(coocMat,threshold,priorLmbda=priorLmbda,
                 numSamples=numSamplesData,deltaSdict=deltaSdict,
                 deltaJdict=deltaJdict,verbose=veryVerbose,
                 meanFieldPriorLmbda=meanFieldPriorLmbda,**kwargs)
        
        if fileStr is not None:
            save(deltaSdict,fileStr+'_deltaSdict.data')
            save(deltaJdict,fileStr+'_deltaJdict.data')
        
        clustersNewLength = np.sum([ len(clusterlist) for clusterlist in clusters.values() ])
        
        if clustersNewLength > clustersOldLength:
            
            if verbose:
                print
                print 'threshold =',threshold
                print 'old number of clusters =',clustersOldLength
                print 'new number of clusters =',clustersNewLength
            
            #m = IsingModel(J,numProcs=numProcs)
            
    #        if saveSamplesAtEveryStep and (fileStr is not None):
    #            # go ahead and take samples to compare with data later
    #            #samples = m.metropolisSamples(1e4,minSize=0)[0]
    #            samples = m.metropolisSamples(numSamplesError)[0]
    #            samplesDict[threshold] = samples
    #            
    #            save(samplesDict,fileStr+'_samplesDict.data')

            # 2.20.2014 calculate individual and pair errors
            #samplesCorrected = m.metropolisSamples(numSamplesError,minSize=0)[0]
            nSkipDefault = 10*self.n
            burninDefault = 100*self.n
            self._multipliers = np.concatenate([J.diagonal(),squareform(zeroDiag(-J))])
            samplesCorrected = self.generate_samples(nSkipDefault,burninDefault,
                                        numSamplesError,'metropolis')
            if minimizeCovariance:
                raise Exception # 3.31.2014 are you sure you want to do this?
                covStdevs = covarianceTildeStdevsFlat(coocMat,
                            numSamplesData,np.diag(coocMat))
                covData = cooccurrences2covariances(coocMat)
                deltaCov = isingDeltaCovTilde(samplesCorrected,covData,np.diag(covData))
                zvals = deltaCov/covStdevs
                
                ell = len(coocMat)
                epsilonp = np.sqrt(np.mean(zvals[:ell]**2))
                epsilonc = np.sqrt(np.mean(zvals[ell:]**2))
                if verbose:
                    print "epsilonp =",epsilonp
                    print "epsilonc =",epsilonc
                epsVals = [epsilonp,epsilonc]
            elif minimizeIndependent: # independent residuals; method of CocMon11
                #
                # 6.27.2014 NOTE!
                # epsilonc is NOT the same as CocMon11.
                # CocMon11 uses connected correlation pij - pi*pj,
                # whereas we use the cooccurrences pij.
                #
                # (future: encapsulate into function?)
                coocStdevs = coocStdevsFlat(coocMat,numSamplesData)
                deltaCooc = isingDeltaCooc(samplesCorrected,coocMat)
                zvals = deltaCooc/coocStdevs
                
                ell = len(coocMat)
                epsilonp = np.sqrt(np.mean(zvals[:ell]**2))
                epsilonc = np.sqrt(np.mean(zvals[ell:]**2))
                if verbose:
                    print "epsilonp =",epsilonp
                    print "epsilonc =",epsilonc
                epsVals = [epsilonp,epsilonc]
            else: # correlated residuals 3.10.2014
                # (future: encapsulate into function?)
                deltaCooc = isingDeltaCooc(samplesCorrected,coocMat)
                # cov = residual covariance
                zvals = np.dot( deltaCooc,U ) / np.sqrt(s)
                coocMatMeanZSq = np.mean( numSamplesData * zvals**2 )
                if verbose:
                    print "coocMatMeanZSq =",coocMatMeanZSq
                epsVals = [coocMatMeanZSq]

            # keep track of mean event size
            meanFightSize = np.mean(np.sum(samplesCorrected,axis=1))
            meanFightSizeList.append(meanFightSize)
            if verbose:
                print "mean event size =",meanFightSize

            # keep track of epsValsList
            epsValsList.append(epsVals)
            #save(epsValsList,fileStr+"_epsValsList.data")

            # 5.21.2014 keep track of thresholds with new clusters,
            # number of clusters, and maximum cluster size
            thresholdList.append(threshold)
            numClustersList.append(clustersNewLength)
            maxClusterSize = max(clusters.keys())
            maxClusterSizeList.append(maxClusterSize)
            if verbose:
                print "max cluster size =",maxClusterSize

            # 5.21.2014
            d = {'meanFightSizeList':meanFightSizeList,
                'epsValsList':epsValsList,
                'thresholdList':thresholdList,
                'numClustersList':numClustersList,
                'maxClusterSizeList':maxClusterSizeList,
            }
            if fileStr is not None:
                save(d,fileStr+"_expansionData.data")

            # keep track of best found so far
            if np.sum(epsVals) < bestEps:
                bestEps = np.sum(epsVals)
                Jbest = J
                clustersBest = clusters
                thresholdBest = threshold

            if np.all( np.array(epsVals) < epsThreshold ):
                stop = True
                if verbose:
                    print
                    print "Using result from threshold =",threshold
                    print
            if threshold < minThreshold:
                stop = True
                
                # go back to best found so far
                if verbose:
                    print "Minimum threshold passed ("+str(minThreshold)+")"
                J = Jbest
                clusters = clustersBest
                threshold = thresholdBest

                if verbose:
                    print
                    print "Using result from threshold =",threshold
                    print

            if maxClusterSize > maxMaxClusterSize: # 5.21.2014
                stop = True
                
                # go back to best found so far
                if verbose:
                    print "Maximum largest cluster size passed ("+str(maxMaxClusterSize)+")"
                J = Jbest
                clusters = clustersBest
                threshold = thresholdBest
                
                if verbose:
                    print
                    print "Using result from threshold =",threshold
                    print

            if stop and bruteForceMin:
                # 3.6.2014 use brute force optimization to get better fit
                thresholdMeanZSq = 1.
                fitAlpha = 20.
                numSamplesBF = int( fitAlpha * numSamplesData / thresholdMeanZSq )
                BFkwargs = {'maxfev':10,
                    'maxnumiter':100,
                    'numProcs':numProcs,
                    'gradFunc':coocJacobianDiagonal,
                    'thresholdMeanZSq':thresholdMeanZSq,
                    'minSize':2,
                    'Jinit':J,
                    'minimizeIndependent':minimizeIndependent,
                    'minimizeCovariance':minimizeCovariance,
                    'priorLmbda':priorLmbda,
                    'numSamples':numSamplesBF,
                    'numFights':numSamplesData}
                if (not minimizeIndependent) and (not minimizeCovariance):
                    coocCov = coocSampleCovariance(data)
                else:
                    coocCov = None
                if verbose:
                    print "Optimizing using findJmatrixBruteForce_CoocMat..."
                J = findJmatrixBruteForce_CoocMat(coocMatObserved,
                                                  coocCov=coocCov,**BFkwargs)

      if retall:
        return J,threshold,clusters,d
      else:
        return J



class RegularizedMeanField(Solver):
    def __init__(self, *args, **kwargs):
        """
        Implementation of regularized mean field method for
        solving the inverse Ising problem, as described in
        Daniels, Bryan C., David C. Krakauer, and Jessica C. Flack. 
        ``Control of Finite Critical Behaviour in a Small-Scale
        Social System.'' Nature Communications 8 (2017): 14301.
        doi:10.1038/ncomms14301
        
        Specific to pairwise Ising constraints.
        
        Params:
        -------
        calc_e (lambda state,params)
            function for computing energies of given state and parameters.  Should take in a 2D state array
            and vector of parameters to compute energies.
        adj (lambda state)
            function for getting all the neighbors of any given state
        calc_de (lambda=None)
            Function for calculating derivative of energy wrt parameters. Takes in 2d state array and index of
            the parameter.
        n_jobs (int=0)
            If 0 no parallel processing, other numbers above 0 specify number of cores to use.
        
        Attributes:
        -----------
        
        Methods:
        --------
        """
        super(RegularizedMeanField,self).__init__(*args,**kwargs)
        self.setup_sampler('metropolis')
    
        # Do I really need this?
        self.samples = np.zeros(self.n)

    def solve(self,samples,
        numSamples=1e5,nSkip=None,seed=0,
        changeSeed=False,numProcs=1,
        numDataSamples=None,minSize=0,
        minimizeCovariance=False,minimizeIndependent=True,
        coocCov=None,priorLmbda=0.,verbose=True,bracket=None,
        numGridPoints=200):
        """
        Varies the strength of regularization on the mean field J to
        best fit given cooccurrence data.
        
        numGridPoints (200) : If bracket is given, first test at numGridPoints
                              points evenly spaced in the bracket interval, then give
                              the lowest three points to scipy.optimize.minimize_scalar
        
        numSamples (1e5)            : 
        nSkip (None)                :
        seed (0)                    :
        changeSeed (False)          :
        numProcs (1)                :
        minSize (0)                 : 3.8.2013 Use a modified model in which
                                      samples with fewer ones than minSize are not
                                      allowed.
        gradDesc (False)            : 5.29.2013 Take a naive gradient descent step
                                      after each LM minimization
        minimizeCovariance (False)  : ** As of 7.20.2017, not currently supported **
                                      6.3.2013 Minimize covariance from emperical
                                      frequencies (see notes); trying to avoid
                                      biases, as inspired by footnote 12 in 
                                      TkaSchBer06
        minimizeIndependent (True)  : ** As of 7.20.2017, minimizeIndependent is 
                                         the only mode currently supported **
                                      2.7.2014 Each <xi> and <xi xj> residual is treated
                                      as independent
        coocCov (None)              : ** As of 7.20.2017, not currently supported **
                                      2.7.2014 Provide a covariance matrix for
                                      residuals.  Should typically be 
                                      coocSampleCovariance(samples).  Only used
                                      if minimizeCovariance and minimizeIndependent
                                      are False.
        priorLmbda (0.)             : ** As of 7.20.2017, not currently implemented **
                                      Strength of noninteracting prior.
        """
        # 7.18.2017 convert input to coocMat
        coocMatData = meanFieldIsing.cooccurrence_matrix((samples+1)/2)
        
        numDataSamples = len(samples)
        
        if coocCov is None:
            coocCov = meanFieldIsing.coocSampleCovariance(samples)
        
        if nSkip is None:
            nSkip = 10*self.n
        
        if changeSeed: seedIter = meanFieldIsing.seedGenerator(seed,1)
        else: seedIter = meanFieldIsing.seedGenerator(seed,0)
        
        if priorLmbda != 0.:
            # 11.24.2014 Need to fix prior implementation
            raise Exception, "priorLmbda is not currently supported"
            lmbda = priorLmbda / numDataSamples

        # 11.21.2014 stuff defining the error model, taken
        #            from findJmatrixBruteForce_CoocMat
        # 3.1.2012 I'm pretty sure the "repeated" line below should have the
        # transpose, but coocJacobianDiagonal is not sensitive to this.  If you
        # use non-diagonal jacobians in the future and get bad behavior you
        # may want to double-check this.
        if minimizeIndependent:
            coocStdevs = meanFieldIsing.coocStdevsFlat(coocMatData,numDataSamples)
            coocStdevsRepeated = scipy.transpose(                                   \
                coocStdevs*scipy.ones((len(coocStdevs),len(coocStdevs))) )
        elif minimizeCovariance:
            raise Exception, "minimizeCovariance is not currently supported"
            empiricalFreqs = scipy.diag(coocMatData)
            covTildeMean = covarianceTildeMatBayesianMean(coocMatData,numDataSamples)
            covTildeStdevs = covarianceTildeStdevsFlat(coocMatData,numDataSamples,
                empiricalFreqs)
            covTildeStdevsRepeated = scipy.transpose(                               \
                covTildeStdevs*scipy.ones((len(covTildeStdevs),len(covTildeStdevs))) )
        else:
            raise Exception, "correlated residuals calculation is not currently supported"
            # 2.7.2014
            if coocCov is None: raise Exception
            cov = coocCov # / numDataSamples (can't do this here due to numerical issues)
                          # instead include numDataSamples in the calculation of coocMatMeanZSq

        # 11.21.2014 for use in gammaPrime <-> priorLmbda
        freqsList = np.diag(coocMatData)
        pmean = np.mean(freqsList)
        
        # 11.21.2014 adapted from findJMatrixBruteForce_CoocMat
        def samples(J):
           seed = seedIter.next()
           #print seed
           #J = unflatten(flatJ,ell,symmetrize=True)
           if minimizeCovariance:
               J = tildeJ2normalJ(J,empiricalFreqs)
           # 7.20.2017 Bryan's old sampler
           #if numProcs > 1:
           #    isingSamples = metropolisSampleIsing_pypar(numProcs,J,
           #                       numSamples,startConfig=None,nSkip=nSkip,
           #                       seed=seed,minSize=minSize)
           #else:
           #    isingSamples = metropolisSampleIsing(J,
           #                     numSamples,startConfig=None,nSkip=nSkip,
           #                     seed=seed,minSize=minSize)
           burninDefault = 100*self.n
           J = J + J.T
           self._multipliers = np.concatenate([J.diagonal(),squareform(meanFieldIsing.zeroDiag(-J))])
           self.generate_samples(nSkip,burninDefault,int(numSamples),'metropolis')
           isingSamples = np.array(self.samples,dtype=float)
           return isingSamples

        # 11.21.2014 adapted from findJMatrixBruteForce_CoocMat
        def func(meanFieldGammaPrime):
            
            # translate gammaPrime prior strength to lambda prior strength
            meanFieldPriorLmbda = meanFieldGammaPrime / (pmean**2 * (1.-pmean)**2)
            
            # calculate regularized mean field J
            J = meanFieldIsing.JmeanField(coocMatData,
                                          meanFieldPriorLmbda=meanFieldPriorLmbda,
                                          numSamples=numDataSamples)

            # sample from J
            isingSamples = samples(J)
            
            # calculate residuals, including prior if necessary
            if minimizeIndependent: # Default as of 4.2.2015
                dc = meanFieldIsing.isingDeltaCooc(isingSamples,coocMatData)/coocStdevs
            elif minimizeCovariance:
                dc = isingDeltaCovTilde(isingSamples,covTildeMean,
                                          empiricalFreqs)/covTildeStdevs
            else:
                dc = meanFieldIsing.isingDeltaCooc(isingSamples,coocMatMean)
                if priorLmbda != 0.:
                    # new prior 3.24.2014
                    # 11.21.2014 oops, I think this should be square-rooted XXX
                    # 11.21.2014 oops, should also apply in minimizeIndependent case XXX
                    freqs = scipy.diag(coocMatData)
                    factor = scipy.outer(freqs*(1.-freqs),freqs*(1.-freqs))
                    factorFlat = aboveDiagFlat(factor)
                    priorTerm = lmbda * factorFlat * flatJ[ell:]**2
                
                dc = np.concatenate([dc,priorTerm])
                
            if verbose:
                print "RegularizedMeanField.solve: Tried "+str(meanFieldGammaPrime)
                print "RegularizedMeanField.solve: sum(dc**2) = "+str(np.sum(dc**2))
                
            return np.sum(dc**2)

        if bracket is not None:
            gridPoints = np.linspace(bracket[0],bracket[1],numGridPoints)
            gridResults = [ func(p) for p in gridPoints ]
            gridBracket = self.bracket1d(gridPoints,gridResults)
            solution = scipy.optimize.minimize_scalar(func,bracket=gridBracket)
        else:
            solution = scipy.optimize.minimize_scalar(func)

        gammaPrimeMin = solution['x']
        meanFieldPriorLmbdaMin = gammaPrimeMin / (pmean**2 * (1.-pmean)**2)
        J = meanFieldIsing.JmeanField(coocMatData,
                                      meanFieldPriorLmbda=meanFieldPriorLmbdaMin,
                                      numSamples=numDataSamples)
        J = J + J.T

        # 7.18.2017 convert J to {-1,1}
        h = -J.diagonal()
        J = -meanFieldIsing.zeroDiag(J)
        self.multipliers = convert_params( h,squareform(J)*2,'11',concat=True )

        return self.multipliers

    # 3.18.2016
    def bracket1d(self,xList,funcList):
        """
        *** Assumes xList is monotonically increasing
        
        Get bracketed interval (a,b,c) with a < b < c, and f(b) < f(a) and f(c).
        (Choose b and c to make f(b) and f(c) as small as possible.)
        
        If minimum is at one end, raise error.
        """
        gridMinIndex = np.argmin(funcList)
        gridMin = xList[gridMinIndex]
        if (gridMinIndex == 0) or (gridMinIndex == len(xList)-1):
            raise Exception, "Minimum at boundary"
        gridBracket1 = xList[ np.argmin(funcList[:gridMinIndex]) ]
        gridBracket2 = xList[ gridMinIndex + 1 + np.argmin(funcList[gridMinIndex+1:]) ]
        gridBracket = (gridBracket1,gridMin,gridBracket2)
        return gridBracket

