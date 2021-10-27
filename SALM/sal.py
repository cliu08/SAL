# -*- coding: utf-8 -*-



def sal(stoalm_dict, loss_dict):
    ''''''



    return stoalm_dict, loss_dict



        loss_dict = {}
        for n, args in loss_args.items():
            try:
                if isinstance(args, tuple) or isinstance(args, list):
                    loss_dict[n] = self.loss_funcs[n](*args)
                    # Add the stoalm algorithm, Dec 29, 18
                    if n != 'ce' and n != 'acc_s' and n != 'acc_t':  # do the lambda updating only for the added loss func
                        if stoalm_dict['fix_weight']:  # fix weight of loss function
                            self.loss_weights[n] = stoalm_dict['loss_func_weight']
                        else:
                            #import time
                            #start_time = time.time()
                            org_loss = loss_dict[n]  # initial loss computed by the called loss function, can be 'mmd', 'assoc', etc.
#                            c_theta = min(0.0, org_loss - stoalm_dict['epsilon'] / np.sqrt(epochi + 1.0))  #  
                            c_theta = org_loss - max(1.0e-5, stoalm_dict['epsilon'] / np.sqrt(epochi + 1.0))  #  
#                            c_theta = org_loss - stoalm_dict['epsilon']
                            #
                            stoalm_dict['add_loss_func'].append(float(loss_dict[n]) )  
                            ## -------------------------------------------------------------------------------------
                            #  update \big V, 
#                            stoalm_dict['bv'].append( float( max( -stoalm_dict['lamb'][-1] / stoalm_dict['rho'][-1], \
#                                                                 c_theta )) )
                            stoalm_dict['bv'].append( float( max( -stoalm_dict['lamb'][-1] / stoalm_dict['rho'][-1], \
                                                                 c_theta )) )
                            ## -------------------------------------------------------------------------------------
                            #  update \lambda
#                            lamb_k = float( min(stoalm_dict['lamb_max'], \
#                                                 max(stoalm_dict['rho'][-1] * c_theta + stoalm_dict['lamb'][-1], 1.0e-5) ) )
#                            lamb_k = min(lamb_k, 1.1 * stoalm_dict['lamb'][-1])  # lambda can not increase too fast
#                            lamb_k = max(lamb_k, 1.0 / 1.1 * stoalm_dict['lamb'][-1])  # lambda can not decrease too fast
                            stoalm_dict['lamb'].append(float( min(stoalm_dict['lamb_max'], \
                                                 max(stoalm_dict['rho'][-1] * c_theta + stoalm_dict['lamb'][-1], 1.0e-5) ) ) )
                            ## -------------------------------------------------------------------------------------
                            #  update \rho or not
                            if len(stoalm_dict['bv']) == 1:
                                stoalm_dict['rho'].append(float(stoalm_dict['rho'][-1])) 
                            elif abs(stoalm_dict['bv'][-1]) <= stoalm_dict['tau'] * abs(stoalm_dict['bv'][-2]):
                                stoalm_dict['rho'].append(max(1.0e-4, float(stoalm_dict['rho'][-1]) / np.sqrt(stoalm_dict['gamma'] ) ))   # for trial
                            else:
                                stoalm_dict['rho'].append( min(1.0e3, float(stoalm_dict['rho'][-1] * stoalm_dict['gamma']) ))
                            ## -------------------------------------------------------------------------------------
                            # -- loss weight, as loss is changed by 'stoalm'
                            self.loss_weights[n] = 1.0
                            ## -------------------------------------------------------------------------------------
                            # -- update the additonal loss, override the MMD measure
                            #    #  min() is used to constrain the absolute value of the addtitional term, 
                            #         -- to avoid to large additional loss term, which could mess the 'ce' loss
                            loss_dict[n] = stoalm_dict['rho'][-1] / 2.0 * max(0.0, \
                                                    (c_theta + stoalm_dict['lamb'][-1] / stoalm_dict['rho'][-1])) ** 2 
                            #print("--- %s seconds ---" % (time.time() - start_time))
                elif isinstance(args, torch.Tensor):
                    loss_dict[n] = args
                else:
                    raise ValueError('Loss args for {} have wrong type. Found {}, expected iterable or tensor'.format(
                        n, type(args)
                    ))
            except Exception as e:
                print('Error in resolving: {}'.format(n))
                raise e
#