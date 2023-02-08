import numpy as np

class Atom():
    
    def __init__(self,r):
        self.r = np.array(r,dtype=float)
        assert self.r.shape == (2,), 'Wrong dimension for position vector, r'
        self.ax = None
        self.artist = None
        self.color = 'C0'
        
    def __repr__(self):
        return 'Atom(r=[{},{}])'.format(self.r[0],self.r[1])

    def __str__(self):
        return 'x={}, y={}, r={}'.format(self.r[0],self.r[1],self.r)
    
    def plot(self,ax):
        
        # create if first time with this Axis
        if self.ax is not ax:
            self.ax = ax
            self.artist = ax.plot(self.r[0],self.r[1],marker='o',
                                  markerfacecolor=self.color,
                                  markeredgecolor='k',
                                 markersize=40)[0]
        else:
            self.artist.set_data(self.r[0],self.r[1])
            self.artist.set_color(self.color)

    def set_color(self,color):
        if isinstance(color,str):
            self.color = color
        elif isinstance(color,int) and 0 < color < 10:
            self.color = f'C{color}'
        else:
            assert False, 'color must be string or an integer below 10'


    
            
class Atoms():
    
    def __init__(self,atoms):
        self.atoms = atoms
        self._current_index = 0
        
    def __iter__(self):
        for atom in self.atoms:
            yield atom
    
    def get_potential_energy(self):
        u = 0
        for atom in self.atoms:
            for other_atom in self.atoms:
                if atom is other_atom:
                    continue
                r = np.linalg.norm(atom.r - other_atom.r)
                u += 4 * ( (1/r)**12 - (1/r)**6 )
        return 0.5 * u
    
    def get_forces(self):
        forces = np.zeros((len(self.atoms),2))
        for atom,f in zip(self.atoms,forces):
            for other_atom in self.atoms:
                if atom is other_atom:
                    continue
                r = atom.r - other_atom.r
                rl = np.linalg.norm(r)
                f += -4 * 6*(rl**6 - 2)/rl**13 * r/rl
        return forces

    def set_positions(self,positions):
        for atom, pos in zip(self.atoms,positions):
            atom.r = pos
