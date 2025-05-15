clear all % clear all variables
clc % clear the command window
tic % start the stopwatch timer

% Change - Ricarda: Setting correct path to the dataset & adding prints for transparency
addr = '/Users/ricardabeck/Dev/Epilepsy/InterIntrapatient/datapreprocessing_Matlab/mitbihdb';
Files=dir(strcat(addr,'/*.mat'));
disp('Files are ');
disp(Files);
disp(Files(1).name);

% AAMI and AAMI2 labling schemes
AAMI_annotations = {'N' 'S' 'V' 'F' 'Q'};
AAMI2_annotations = {'N' 'S' 'V_hat' 'Q'};

index = 1;
beat_len = 280;
n_cycles = 0;
featuresSeg = [];
groupN = [];
groupS = [];
groupV = [];
groupF = [];
groupQ = [];
N_class = 0;S_class=0;V_class=0;F_class=0;Q_class=0;

% iterate over all files
for i=1:length(Files)
    
    % counter
    disp(' ----------------- Patient -----------------');

    % extract file name 
    [pathstr,name,ext] = fileparts(Files(i).name);

    % load the ECG signal
    [tm,ecgsig,ann,Fs,sizeEcgSig,timeEcgSig] = loadEcgSig([addr filesep name]);
    % take the first element
    nsig = 1;
    signal = ecgsig(nsig,:);
    
    % peak detection
    rPeaks  = cell2mat(ann(3))+1;
    n_cycles = n_cycles + length(rPeaks);
    rPeaks = double(rPeaks);
    peaks = qsPeaks(signal, rPeaks, Fs);
    tpeaks = peaks(:,7);

    annots_list = ['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q'];
    annot  = cell2mat(ann(4));
    indices  = ismember(rPeaks,peaks(:,4));
    annot = annot(indices);

    seg_values = {};
    seg_labels =[];
    ind_seg = 1;

    % normalize
    signal = normalize(signal);

    % Test Ricarda
    N_count = 0;
    S_count = 0;
    V_count = 0;
    F_count = 0;
    Q_count = 0;

    disp('Count of annotations');
    disp(length(annot));

    for ind = 1:length(annot)
        if ~ismember(annot(ind),annots_list)
            continue;
        end

        N_g = ['N', 'L', 'R', 'e', 'j'];%0
        S_g = ['A', 'a', 'J', 'S'];%1
        V_g = ['V', 'E'];%2
        F_g = ['F'];%3
        Q_g = [' /', 'f', 'Q'];%4


        if(ismember(annot(ind),N_g))
            lebel = 'N';
            N_count = N_count + 1;
        elseif(ismember(annot(ind),S_g))
            lebel = 'S';
            S_count = S_count + 1;
        elseif(ismember(annot(ind),V_g))
            lebel = 'V';
            V_count = V_count + 1;
        elseif(ismember(annot(ind),F_g))
            lebel = 'F';
            F_count = F_count + 1;
        elseif(ismember(annot(ind),Q_g))
            lebel = 'Q';
            Q_count = Q_count + 1;
        else
            throw("No label! :(")

        end

        if ind==1

            seg_values{ind_seg} = signal(1:tpeaks(ind)-1)';
            t_sig = imresize(seg_values{ind_seg}(1:min(Fs,length(seg_values{ind_seg}))), [beat_len 1]);
            seg_values{ind_seg} = t_sig;
            seg_labels(ind_seg) = lebel;
            ind_seg = ind_seg+1;
            continue;
        end
        t_sig = imresize(signal(tpeaks(ind-1):tpeaks(ind)-1)', [beat_len 1]);
        seg_values{ind_seg} = t_sig;
        seg_labels(ind_seg) = lebel;
        ind_seg = ind_seg+1;

        % Test Ricarda
        if ind == 1
            disp(seg_labels);
        end

    end

    % Test Ricarda
    disp('N: '); disp(N_count);
    disp('S: '); disp(S_count);
    disp('V: '); disp(V_count);
    disp('F: '); disp(F_count);
    disp('Q: '); disp(Q_count);

    % store values and labels per patient
    s2s_mitbih(i).seg_values = seg_values';
    s2s_mitbih(i).seg_labels = char(seg_labels);

    % group N:0
    % N = N, L, R, e, j
    N_inds = find(annot=='N');
    N_inds = [N_inds;find(annot=='L')];
    N_inds = [N_inds;find(annot=='R')];
    N_inds = [N_inds;find(annot=='e')];
    N_inds = [N_inds;find(annot=='j')];
    N_class = N_class + length(N_inds);

    % group S:1
    % S = A, a, J, S
    S_inds = find(annot=='S');
    S_inds = [S_inds;find(annot=='A')];
    S_inds = [S_inds;find(annot=='a')];
    S_inds = [S_inds;find(annot=='J')];
    S_class = S_class + length(S_inds);

    % group V:2
    % V = V, E
    V_inds = find(annot=='V');
    V_inds = [V_inds;find(annot=='E')];
    V_class = V_class + length(V_inds);

    % group F:3
    % F = F
    F_inds = find(annot=='F');
    F_class = F_class + length(F_inds);

    % group Q:4
    % Q = /, f, Q
    Q_inds = find(annot=='/');
    Q_inds = [Q_inds;find(annot=='f')];
    Q_inds = [Q_inds;find(annot=='Q')];
    Q_class = Q_class + length(Q_inds);

    % Test Ricarda
    % disp('Q only + Total cummulative');
    % length(Q_inds)
    % F_class+N_class+Q_class+S_class+V_class

end

% Print the number of values present in each class
F_class
N_class
Q_class
S_class
V_class
F_class+N_class+Q_class+S_class+V_class

save s2s_mitbih_aami.mat s2s_mitbih
toc
disp('Successfully generated :)')
