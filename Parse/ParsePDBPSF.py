
import numpy as np
import os.path
import re
import math

def next_power_of_2(x):
    return 1 if x == 0 else int(2**math.ceil(math.log(x,2)))

# Read PSF

fname = '1dgb_autopsf.psf'
with open(fname) as f:
    PSF = f.readlines()

#PSF = [x.strip() for x in PSF]
## Read bonds

get_bonds_from_now_on = 0
get_types_from_now_on = 0

for x in PSF:
    if get_bonds_from_now_on:
        #print(idx)
        temp_bonds = re.findall('\d+', x)
        temp_bonds = map(int, temp_bonds)
        while (idx < NBOND):
            bonds[idx][:] = temp_bonds[0:2]
            #print(bonds[idx][:])
            idx = idx + 1
            if (len(temp_bonds) > 2):
                temp_bonds[0:2] = []
            else:
                break
    if get_types_from_now_on:
        idx = int(re.search(r'\d+', x).group())
#        print(idx)
#        print(x)
        if x[24] == 'H':
            Ele[idx-1] = 0
            #print('Hydrogen')
        elif x[24] == 'C':
            Ele[idx-1] = 1 
        elif x[24] == 'N':
            Ele[idx-1] = 2 
        elif x[24] == 'O':
            Ele[idx-1] = 3 
        elif x[24] == 'S':
            Ele[idx-1] = 4 
        elif x[24:26] == 'Fe':
            Ele[idx-1] = 5
#        print(Ele[idx-1]) 
        if (idx == NATOM):
            print('Recorded all atoms.')
            get_types_from_now_on = 0

    if '!NBOND:' in x:
        print('NBONDS found.')
        print(x)
        print(re.search(r'\d+', x).group())
        NBOND = int(re.search(r'\d+', x).group())
        print('There are {:d} bonds.'.format(NBOND))
        get_bonds_from_now_on = 1
        idx = 0
        bonds = np.zeros((NBOND,2),dtype=int)

    if '!NATOM' in x:
        NATOM = int(re.search(r'\d+', x).group())
        print(NATOM)
        atoms = np.zeros((NATOM,3))
        Ele = np.zeros((NATOM,1),dtype=int)
        get_types_from_now_on = 1

print(bonds)
bonds = bonds.flatten()
print(Ele)


# Read PDB
fname = '1dgb_autopsf.pdb'
with open(fname) as f:
    PDB = f.readlines()

PDB = [x.strip().split() for x in PDB]

## Read coordinates
for x in PDB:
    if x[0] == 'ATOM':
        atoms[int(x[1])-1][:] = x[6:9]

print(atoms)


HC = 0
HN = 0
HO = 0
HS = 0
num_ele = 5

for idx, atom in enumerate(Ele):
    if atom == 0:
#        print('Idx is {:d}'.format(idx))
        atom_H = bonds.tolist().index(idx+1)
#        print('atom_H index is {:d}'.format(atom_H))
        if (atom_H % 2 == 0):
            atom_X = atom_H + 1
        else:
            atom_X = atom_H - 1
#        print('atom_X index is {:d}'.format(atom_X))
        if (Ele[bonds[atom_X]-1] == 1):
            A = 'Carbon'
            HC = HC + 1
        elif (Ele[bonds[atom_X]-1] == 2):
            A = 'Nitrogen'
            HN = HN + 1
        elif (Ele[bonds[atom_X]-1] == 3):
            A = 'Oxygen'
            HO = HO + 1
        elif (Ele[bonds[atom_X]-1] == 4):
            A = 'Sulfur'
            HS = HS + 1

#        print('Corresponding heavy atom is {:d} ({:s})'.format(Ele[bonds[atom_X]-1],A))
        Ele[idx] = Ele[idx] + num_ele + Ele[bonds[atom_X]-1]

#print(Ele.tolist())

## Print things

with open('mol_param.hh','w') as f:
    f.write('\n')
    f.write('extern int Ele[{:d}];\n'.format(NATOM))
    f.write('extern int num_atom;\n')
    f.write('extern int num_atom2;\n')

with open('mol_param.cu','w') as f:
    f.write('\n#include "mol_param.hh"\n\n')
    f.write('int num_atom = {:d};\n'.format(NATOM))
    f.write('int num_atom2 = {:d};\n\n'.format(next_power_of_2(NATOM)))
    f.write('int Ele[{:d}] = {{'.format(NATOM))
    f.write(', '.join(map(str,Ele.flatten())))
    f.write('};\n')
 
with open('coord_ref.hh','w') as f:
    f.write('\nextern float coord_ref[{:d}];\n'.format(3 * NATOM))
    f.write('extern float coord_init[{:d}];\n'.format(3 * NATOM))
    
with open('coord_ref.cu','w') as f:
    f.write('\n#include "coord_ref.hh"\n\n')
    f.write('float coord_ref[{:d}] = {{'.format(3 * NATOM))
    f.write(', '.join(map(str,atoms.flatten())))
    f.write('};\n')
    f.write('float coord_init[{:d}] = {{'.format(3 * NATOM))
    f.write(', '.join(map(str,atoms.flatten())))
    f.write('};\n')


