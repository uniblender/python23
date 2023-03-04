import numpy as np

def sanitize_color(color):
    if isinstance(color,str):
        pass
    elif isinstance(color,int) and 0 < color < 10:
        color = f'C{color}'
    else:
        assert False, 'color must be string or an integer below 10'
    return color

def sanitize_position(pos):
    pos = np.array(pos,dtype=float)
    assert pos.shape == (2,), 'position must be 2D'
    return pos

t4plot = np.linspace(0, 2*np.pi, 30)
x4plot = 2**(1/6)/2*np.cos(t4plot)
y4plot = 2**(1/6)/2*np.sin(t4plot)

class Atom():
    
    def __init__(self,r,color='C0'):
        self.r = sanitize_position(r)
        assert self.r.shape == (2,), 'Wrong dimension for position vector, r'
        self.ax = None
        self.artist = None
        self.color = sanitize_color(color)
        
    def __repr__(self):
        return 'Atom(r=[{},{}], color={})'.format(self.r[0],self.r[1],self.color)

    def __str__(self):
        return 'x={}, y={}, r={}, color={}'.format(self.r[0],self.r[1],self.r, self.color)
    
    def plot(self,ax):
        
        # create if first time with this Axis
        if self.ax is not ax:
            self.ax = ax
            #self.artist = ax.plot(self.r[0],self.r[1],marker='o',
            #                      markerfacecolor=self.color,
            #                      markeredgecolor='k',
            #                     markersize=40)[0]
            self.artist = ax.fill(x4plot + self.r[0],
                                  y4plot + self.r[1],
                                  facecolor = self.color,
                                  edgecolor = 'k')[0]
        else:
            xy = np.array([x4plot + self.r[0], y4plot + self.r[1]]).T
            self.artist.set_xy(xy)

    def get_position(self):
        return self.r

    def set_position(self,pos):
        self.r = sanitize_position(pos)
        if self.artist is not None:
            xy = np.array([x4plot + self.r[0], y4plot + self.r[1]]).T
            self.artist.set_xy(xy)
            
    def set_color(self,color):
        self.color = sanitize_color(color)
        if self.artist is not None:
            self.artist.set_markerfacecolor(self.color)


    
            
class Atoms():
    
    def __init__(self,atoms):
        self.atoms = atoms
        
    def __iter__(self):
        for atom in self.atoms:
            yield atom

    def __getitem__(self, item):
        return self.atoms[item]
            
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

    def get_positions(self):
        positions = np.zeros((len(self.atoms),2))
        for atom,pos in zip(self.atoms,positions):
            pos += atom.r
        return positions

    def set_positions(self, positions, update_plot=True):
        for atom, pos in zip(self.atoms,positions):
            if update_plot:
                atom.set_position(pos)
            else:
                atom.r = pos

    def get_atomic_potential(self, xs, ys):
        e0 = self.get_potential_energy()

        shape = np.array(xs).shape
        xsflat = np.array(xs).flatten()
        ysflat = np.array(ys).flatten()
        zsflat = []
        
        for x, y in zip(xsflat, ysflat):
            p = [x,y]
            extra_atom = Atom(p)
            atoms_plus_one = Atoms([atom for atom in self] + [extra_atom])
            zsflat.append(atoms_plus_one.get_potential_energy() - e0)
        zs = np.reshape(np.array(zsflat),shape)
        return zs
