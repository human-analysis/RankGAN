# visualizer.py

from os import sys
import visdom as visdom
import numpy as np
import torch
from datasets import transforms

from plotly import tools
import plotly.graph_objs as go

from utils import plotlify, _debuginfo


class Visualizer:
    def __init__(self, port, env, title):

        self.keys = []
        self.values = {}
        self.env = env
        self.viz = visdom.Visdom(port=port, env=env)
        self.iteration = 0
        self.title = title

    def register(self, modules):
        # here modules are assumed to be a dictionary
        for key in modules:
            self.keys.append(key)
            self.values[key] = {}
            self.values[key]['dtype'] = modules[key]['dtype']
            self.values[key]['vtype'] = modules[key]['vtype']
            self.values[key]['win'] = modules[key]['win'] \
                                        if 'win' in modules[key].keys() \
                                        else None
            self.values[key]['layout'] = modules[key]['layout'] \
                    if 'layout' in modules[key].keys() \
                    else {'windows': [key], 'id': 0}

            if modules[key]['vtype'] == 'plot':
                self.values[key]['value'] = []
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                # Edit the layout
                layout = dict(
                    title=key,
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title=key),
                    )
                fig = dict(data=data, layout=layout)
                self.values[key]['win'] = self.viz._send(
                    plotlify(fig, env=self.env, win=self.values[key]['win']))
            elif modules[key]['vtype'] in ('image', 'images'):
                self.values[key]['value'] = None
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

    def update(self, modules):
        for key in modules:
            if self.values[key]['dtype'] == 'scalar':
                self.values[key]['value'].append(modules[key])
            elif self.values[key]['dtype'] in ('image', 'images'):
                self.values[key]['value'] = modules[key]
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

        for key in self.keys:
            if self.values[key]['vtype'] == 'plot':
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                ) for i in range(len(self.values[key]['layout']['windows']))]
                try:
                    data[self.values[key]['layout']['id']] = go.Scatter(
                        x=np.array([self.iteration]).tolist(),
                        y=np.array([self.values[key]['value'][-1]]).tolist(),
                    )
                except Exception as e:
                    import pdb; pdb.set_trace()  # breakpoint 344a72de //

                fig = dict(data=data, append=True)
                self.viz._send(
                    plotlify(fig, env=self.env,
                             win=self.values[key]['win']), endpoint='update')
            elif self.values[key]['vtype'] == 'image' and self.values[key]['value'] is not None:
                if torch.is_tensor(self.values[key]['value']):
                    temp = self.values[key]['value'].numpy() if torch.is_tensor(self.values[key]['value']) else self.values[key]['value']
                    for i in range(temp.shape[0]):
                        temp[i] = temp[i] - temp[i].min()
                        temp[i] = temp[i] / temp[i].max()
                else:
                    temp = self.values[key]['value']
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
                else:
                    self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
            elif self.values[key]['vtype'] == 'images' and self.values[key]['value'] is not None:
                if torch.is_tensor(self.values[key]['value']):
                    temp = self.values[key]['value'].numpy() if torch.is_tensor(self.values[key]['value']) else self.values[key]['value']
                    for i in range(temp.shape[0]):
                        for j in range(temp.shape[1]):
                            temp[i][j] = temp[i][j] - temp[i][j].min()
                            temp[i][j] = temp[i][j] / temp[i][j].max()
                else:
                    temp = self.values[key]['value']
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.images(
                        temp, nrow=11,  # TODO: needs to be automated later
                        win=self.values[key]['win'],
                        opts=dict(title=key,
                                  caption=self.iteration)
                    )
                else:
                    self.viz.images(
                        temp, nrow=11,  # TODO: needs to be automated later
                        win=self.values[key]['win'],
                        opts=dict(title=key,
                                  caption=self.iteration)
                    )
            else:
                pass
                # raise Exception('Visualization type not supported, please '
                                # 'update the visualizer plugin and rerun !!')
        self.iteration = self.iteration + 1


class HourGlassVisualizer:
    def __init__(self, port, env, title):

        self.keys = []
        self.values = {}
        self.env = env
        self.viz = visdom.Visdom(port=port, env=env)
        self.iteration = 0
        self.title = title
        # self.toColorHeatmap = transforms.ToColorHeatmap()
        # self.toLandmarks = transforms.ToLandmarks()

    def register(self, modules, reset=True):
        # here modules are assumed to be a dictionary
        for key in modules:
            self.keys.append(key)
            self.values[key] = {}
            self.values[key]['dtype'] = modules[key]['dtype']
            self.values[key]['vtype'] = modules[key]['vtype']
            self.values[key]['win'] = modules[key]['win'] \
                                        if 'win' in modules[key].keys() \
                                        else None
            if modules[key]['vtype'] == 'plot':
                self.values[key]['layout'] = modules[key]['layout'] \
                    if 'layout' in modules[key].keys() \
                    else {'windows': [key], 'id': 0}
                self.values[key]['value'] = []
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                    mode='lines',
                    name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                # Edit the layout
                layout = dict(
                    title=key,
                    xaxis=dict(title='Epoch'),
                    yaxis=dict(title=key),
                    )
                fig = dict(data=data, layout=layout)
                if reset:
                    self.values[key]['win'] = self.viz._send(
                        plotlify(fig, env=self.env, win=self.values[key]['win']))

            elif modules[key]['vtype'] == 'scatter':
                self.values[key]['layout'] = modules[key]['layout'] \
                    if 'layout' in modules[key].keys() \
                    else {'windows': [key], 'id': 0}
                self.values[key]['value'] = []
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                # Edit the layout
                layout = dict(
                    title=key,
                    xaxis=dict(title='x'),
                    yaxis=dict(title='y'),
                    )
                fig = dict(data=data, layout=layout)
                if reset:
                    self.values[key]['win'] = self.viz._send(
                        plotlify(fig, env=self.env, win=self.values[key]['win']))

            elif modules[key]['vtype'] == 'histogram':
                self.values[key]['value'] = np.array([])
                self.values[key]['layout'] = modules[key]['layout'] \
                    if 'layout' in modules[key].keys() \
                    else {'windows': [key], 'id': 0}
                self.values[key]['value'] = []
                # Create traces
                data = [go.Histogram(
                    x=[],
                    histnorm='probability'
                    # name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                # Edit the layout
                layout = dict(
                    title=key,
                    xaxis=dict(title='Epoch'),
                    )
                fig = dict(data=data, layout=layout)
                if reset:
                    self.values[key]['win'] = self.viz._send(
                        plotlify(fig, env=self.env, win=self.values[key]['win']))


            elif modules[key]['vtype'] in ('image', 'images',
                                           'heatmap', 'heatmaps'):
                self.values[key]['value'] = None
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

    def reset(self, key):
        self.values[key]['value'] = np.array([])

    def update(self, modules):
        for key in modules:
            if self.values[key]['dtype'] == 'scalar':
                self.values[key]['value'].append(modules[key])
            elif self.values[key]['dtype'] == 'vector':
                if len(self.values[key]['value']) == 0:
                    self.values[key]['value'] = modules[key].numpy()
                else:
                    self.values[key]['value'] = np.concatenate((self.values[key]['value'], modules[key].numpy()), axis=0)
            elif self.values[key]['dtype'] in ('image', 'images',
                                               'heatmap'):
                self.values[key]['value'] = modules[key]
            elif self.values[key]['dtype'] in ('heatmaps', ):
                input, label, output = modules[key]
                images = list()

                # output_landmarks = self.toLandmarks(output)
                for i in range(len(input)):
                    images.append(input[i])
                    images.append(
                        0.5 * input[i] +
                        0.5 * self.toColorHeatmap(
                            label[i].sum(dim=0),
                            input[i].size()[1:]))
                    for hm_i in range(len(output[-1].cpu().data[i])):
                        images.append(
                            0.5 * input[i] +
                            0.5 * self.toColorHeatmap(
                                output[-1].cpu().data[i][hm_i],
                                input[i].size()[1:]))
                self.values[key]['value'] = torch.stack(images)
            else:
                raise Exception('Data type not supported, please update the '
                                'visualizer plugin and rerun !!')

        for key in self.keys:
            if self.values[key]['vtype'] == 'plot':
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                ) for i in range(len(self.values[key]['layout']['windows']))]
                data[self.values[key]['layout']['id']] = go.Scatter(
                    x=np.array([self.iteration]).tolist(),
                    y=np.array([self.values[key]['value'][-1]]).tolist(),
                )
                fig = dict(data=data, append=True)
                self.viz._send(
                    plotlify(fig, env=self.env,
                             win=self.values[key]['win']), endpoint='update')

            elif self.values[key]['vtype'] == 'scatter':
                # Create traces
                data = [go.Scatter(
                    x=[],
                    y=[],
                ) for i in range(len(self.values[key]['layout']['windows']))]
                values = self.values[key]['value']
                data[self.values[key]['layout']['id']] = go.Scatter(
                    x=values[:,0].tolist(),
                    y=values[:,1].tolist(),
                )
                fig = dict(data=data, append=True)
                self.viz._send(
                    plotlify(fig, env=self.env,
                             win=self.values[key]['win']), endpoint='update')

            elif self.values[key]['vtype'] == 'histogram':
                # Create traces
                # print(self.values[key]['value'].shape)
                data = [go.Histogram(
                    x=[],
                    histnorm='probability'
                    # name=self.values[key]['layout']['windows'][i]
                ) for i in range(len(self.values[key]['layout']['windows']))]
                data[self.values[key]['layout']['id']] = go.Histogram(
                    x=self.values[key]['value'].numpy().tolist(),
                    histnorm='probability'
                )
                layout = dict(
                    title=key,
                    xaxis=dict(title='Epoch'),
                    )
                fig = dict(data=data, layout=layout)
                self.values[key]['win'] = self.viz._send(
                    plotlify(fig, env=self.env, win=self.values[key]['win']))
            elif self.values[key]['vtype'] == 'image':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    temp[i] = temp[i] - temp[i].min()
                    temp[i] = temp[i] / temp[i].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
                else:
                    self.viz.image(
                        temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, caption=self.iteration)
                    )
            elif self.values[key]['vtype'] == 'images':
                temp = self.values[key]['value'].numpy()
                for i in range(temp.shape[0]):
                    for j in range(temp.shape[1]):
                        temp[i][j] = temp[i][j] - temp[i][j].min()
                        temp[i][j] = temp[i][j] / temp[i][j].max()
                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.images(
                        temp, nrow=11, # FIXME: needs to be automated later
                        win=self.values[key]['win'],
                        opts=dict(title=key,
                                  caption=self.iteration)
                    )
                else:
                    self.viz.images(
                        temp, nrow=11, # FIXME: needs to be automated later
                        win=self.values[key]['win'],
                        opts=dict(title=key,
                                  caption=self.iteration)
                    )
            elif self.values[key]['vtype'] == 'heatmap':
                temp = self.values[key]['value'].numpy().astype(np.float64)    #.numpy().astype(np.float64)

                if self.iteration == 0:
                    self.values[key]['win'] = self.viz.surf(
                        X=temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, colormap='Hot',
                                  caption=self.iteration)
                    )
                else:
                    self.viz.surf(
                        X=temp,
                        win=self.values[key]['win'],
                        opts=dict(title=key, colormap='Hot',
                                  caption=self.iteration)
                    )
            elif self.values[key]['vtype'] == 'heatmaps':
                fig = tools.make_subplots(rows=2, cols=9)

                X = np.squeeze(X)
                assert X.ndim == 2, 'X should be two-dimensional'

                opts = {} if opts is None else opts
                opts['xmin'] = opts.get('xmin', X.min())
                opts['xmax'] = opts.get('xmax', X.max())
                opts['colormap'] = opts.get('colormap', 'Viridis')
                _assert_opts(opts)

                data = [{
                    'z': X.tolist(),
                    'cmin': opts['xmin'],
                    'cmax': opts['xmax'],
                    'type': stype,
                    'colorscale': opts['colormap']
                }]

                fig.append_trace(trace1, 1, 1)
                fig.append_trace(trace2, 1, 2)

                fig['layout'].update(height=600, width=600,
                                     title='i <3 subplots')

            else:
                raise Exception('Visualization type not supported, please '
                                'update the visualizer plugin and rerun !!')
        self.iteration = self.iteration + 1
