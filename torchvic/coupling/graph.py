"""The dependency graph of a coupled run.

Nodes are the enabled modules; an edge producer -> consumer is a connection
with its lag (in the consumer's exchange intervals). A lag-0 edge means the
consumer reads what the producer proposes for the same interval, so the
producer must compute first; an edge with lag >= 1 reads accepted values
from earlier intervals and imposes no order. A cycle of lag-0 edges cannot
be ordered: it is refused unless every connection in it declares
`iterate`, in which case its members are computed together until their
exchanged values stop changing (Runtime).
"""
from .spec import CouplingError


def _sccs(nodes, edges):
    """Tarjan's strongly connected components, in reverse topological
    order of the condensation."""
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)

    def visit(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in adj[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for n in nodes:
        if n not in index:
            visit(n)
    return out


class DependencyGraph(object):
    """`modules`: ordered module names; `connections`: dicts with 'from'
    and 'to' ("module.var"), 'lag' and 'iterate'; `declared`: {module:
    names of modules it declares it needs}."""

    def __init__(self, modules, connections, declared=None):
        self.modules = list(modules)
        self.connections = list(connections)
        problems = []
        for m, deps in (declared or {}).items():
            for d in deps:
                if d not in self.modules:
                    problems.append('%s needs %r, which is not an enabled '
                                    'module (missing dependency)' % (m, d))
        if problems:
            raise CouplingError('; '.join(problems))
        edges0 = []
        self.iterative = {}
        for c in self.connections:
            a, b = c['from'].split('.')[0], c['to'].split('.')[0]
            if c.get('lag', 0) == 0:
                edges0.append((a, b))
        # a declared dependency is an ordering edge: the module it needs
        # computes first (a connector that stages inputs for another's
        # shared step, e.g. a reservoir for routing, is a dependency of it)
        dep_edges = [(d, m) for m, deps in (declared or {}).items()
                     for d in deps if (d, m) not in edges0]
        comps = _sccs(self.modules, edges0 + dep_edges)
        comps.reverse()                     # producers first
        self.groups = []
        for comp in comps:
            comp = sorted(comp, key=self.modules.index)
            cyclic = len(comp) > 1 or (comp[0], comp[0]) in edges0
            if cyclic and not any(
                    c.get('lag', 0) == 0
                    and c['from'].split('.')[0] in comp
                    and c['to'].split('.')[0] in comp
                    for c in self.connections):
                raise CouplingError(
                    'declared dependencies form a cycle among %s' % comp)
            if cyclic:
                inside = [c for c in self.connections
                          if c.get('lag', 0) == 0
                          and c['from'].split('.')[0] in comp
                          and c['to'].split('.')[0] in comp]
                if not all(c.get('iterate') for c in inside):
                    raise CouplingError(
                        'dependency cycle without lag among %s: give one of '
                        'the connections a lag >= 1, or declare iterate on '
                        'every lag-0 connection of the cycle (%s)'
                        % (comp, ', '.join('%s -> %s' % (c['from'], c['to'])
                                           for c in inside)))
                it = [c['iterate'] for c in inside]
                self.iterative[tuple(comp)] = dict(
                    max_iter=max(int(x.get('max_iter', 10)) for x in it),
                    tol=min(float(x.get('tol', 1e-9)) for x in it))
            self.groups.append(comp)

    @property
    def order(self):
        return [m for g in self.groups for m in g]

    def upstream(self, module):
        return [c for c in self.connections
                if c['to'].split('.')[0] == module]
