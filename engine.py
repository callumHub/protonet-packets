from tqdm import tqdm
import torch
class Engine(object):
    '''
    Engine class entirely from Jake Snells prototypical network classifier
    '''
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        losses = []
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])
        display_loss = 10.0
        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])
            # REMOVED TQDM FOR IRACE
            pbar = tqdm(state['loader'])
            # For no tqdm, replace pbar with: state["loader"]
            for sample in pbar:
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'], state['batch'])
                self.hooks['on_forward'](state)
                if torch.isnan(loss):
                    raise RuntimeError('Loss is NaN, episode: ', state['batch'])
                display_loss = loss.item()
                loss.backward()
                self.hooks['on_backward'](state)
                torch.nn.utils.clip_grad_value_(state['model'].parameters(), 100) # NEW

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)
                pbar.set_description("Epoch {:d} train, loss {:.5f}".format(state['epoch'] + 1, display_loss))
                losses.append(display_loss)


            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
        import numpy as np
        np.save(arr=np.asarray(losses), file="../runs/train_losses/losses.npy")