
% create mitbih datasets: download all records and convert the records into .info,.mat and .txt(for annotations)
% before doing it, you have to install the open-source WFDB Software Package available at
% http://physionet.org/physiotools/wfdb.shtml 

% download the whole database but files have .dat and .atr extentions
rc=physionetdb('mitdb',1); 

% input
% output .info, .hea, .mat and.txt files 

record_list = physionetdb('mitdb');

%path_to_exes = 'mcode\nativelibs\macosx\bin';
%path_to_save_records = 'mitbihdb';

path_to_exes = '/Users/ricardabeck/Dev/Epilepsy/InterIntrapatient/datapreprocessing_Matlab/mcode/nativelibs/macosx/bin';
path_to_save_records = '/Users/ricardabeck/Dev/Epilepsy/InterIntrapatient/datapreprocessing_Matlab/mitbihdb';

% path_to_exes = 'C:\my_files\ECG_research\mcode\nativelibs\windows\bin';
% path_to_save_records = 'C:\my_files\ECG_dataset\MIT-BIH\mitbihdb_code';

%if ~exist(path_to_save_records, 'dir')
%    mkdir(path_to_save_records);
%end

cd(path_to_save_records);
disp(dir);

tic
for i=1:length(record_list)

    %record_id = num2str(record_list(i)); % Wandelt den Datensatz in einen String um
    
    command_annot = char(strcat(path_to_exes, filesep, 'rdann.exe -r mitdb/', record_list(i), ' -a atr -v >', record_list(i), 'm.txt'));
    system (command_annot);
    command_mat_info_ = char(strcat(path_to_exes, filesep, 'wfdb2mat.exe -r mitdb/', record_list(i), ' >', record_list(i), 'm.info'));
    system (command_mat_info_);
    
%     system('C:\my_files\ECG_research\mcode\nativelibs\windows\bin\wfdb2mat.exe -r 100s -f 0 -t 10 >100sm.info')
%     system('C:\my_files\ECG_research\mcode\nativelibs\windows\bin\rdann.exe -r mitdb/100 -a atr >100.txt')
    
end
toc
disp('Successfully generated :)')

