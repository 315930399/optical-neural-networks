function out = model
%
% Untitled.m
%
% Model exported on Mar 16 2019, 11:35 by COMSOL 5.3.0.223.

import com.comsol.model.*
import com.comsol.model.util.*

model = ModelUtil.create('Model');

model.modelPath(['C:\Users\Ying Yue\Desktop\' native2unicode(hex2dec({'59' '27'}), 'unicode')  native2unicode(hex2dec({'52' '1b'}), 'unicode') '\' native2unicode(hex2dec({'4e' 'ff'}), 'unicode')  native2unicode(hex2dec({'77' '1f'}), 'unicode') ]);

model.comments([native2unicode(hex2dec({'67' '2a'}), 'unicode')  native2unicode(hex2dec({'54' '7d'}), 'unicode')  native2unicode(hex2dec({'54' '0d'}), 'unicode') '\n\n']);

model.component.create('comp1', true);

model.component('comp1').geom.create('geom1', 3);

model.component('comp1').mesh.create('mesh1');

model.component('comp1').physics.create('emw', 'ElectromagneticWaves', 'geom1');

data = importdata('data.txt');
digit = importdata('digit0.txt');

a = size(data);
for  i = 1:a(1)
    da = data(i,:);
    name1 = ['data', num2str(i)];
    model.component('comp1').geom('geom1').create(name1, 'Block');
    model.component('comp1').geom('geom1').feature(name1).set('size', 100*da(1:3));
    model.component('comp1').geom('geom1').feature(name1).set('pos', 100*da(4:6));
    model.component('comp1').geom('geom1').run(name1);
    di = digit(i,:);
    name2 = ['digit', num2str(i)];
    model.component('comp1').geom('geom1').create(name2, 'Block');
    model.component('comp1').geom('geom1').feature(name2).set('size', 100*di(1:3));
    model.component('comp1').geom('geom1').feature(name2).set('pos', 100*di(4:6));
    model.component('comp1').geom('geom1').run(name2);
end

out = model;
