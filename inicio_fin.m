% Cargar señal de voz
[signal, fs] = audioread('fa.wav'); 

% Parámetros de ventana
frame_length = round(0.02 * fs);  % 20 ms
hop_length = round(0.01 * fs);    % 10 ms

num_frames = floor((length(signal) - frame_length)/hop_length) ;
zcr = zeros(1, num_frames);
energy = zeros(1, num_frames);

% Calcular ZCR y energía
for i = 1:num_frames
    start_idx = (i-1)*hop_length + 1;
    frame = signal(start_idx : start_idx + frame_length - 1);
    
    crossings = sum(abs(diff(sign(frame)))) / 2;
    zcr(i) = crossings / frame_length;
    
    energy(i) = sum(frame.^2) / frame_length;
end

% Umbrales
zcr_threshold = 0.08 * max(zcr);
energy_threshold = 0.03 * max(energy);

% Detección de voz (inicio y fin)
voice_flags = (zcr > zcr_threshold) & (energy > energy_threshold);

% Buscar primer y último frame con voz
first_voice_frame = find(voice_flags, 1, 'first');
last_voice_frame = find(voice_flags, 1, 'last');

if isempty(first_voice_frame) || isempty(last_voice_frame)
    disp('No se detectó voz en la señal.');
    return;
end

% Convertir a índices de muestra
start_sample = (first_voice_frame - 1)*hop_length + 1;
end_sample = (last_voice_frame - 1)*hop_length + frame_length;

% Asegurar que no salimos del límite
end_sample = min(end_sample, length(signal));

% Recortar señal
signal_trimmed = signal(start_sample:end_sample);

% Reproducir y guardar (opcional)
sound(signal_trimmed, fs);
audiowrite('voz_recortada.wav', signal_trimmed, fs);

% Visualización
t = (0:length(signal)-1)/fs;
figure;
subplot(2,1,1);
plot(t, signal); hold on;
xline(start_sample/fs, 'g--', 'Inicio detectado');
xline(end_sample/fs, 'r--', 'Fin detectado');
title('Señal original con detección de inicio y fin de palabra');

subplot(2,1,2);
t_trimmed = (0:length(signal_trimmed)-1)/fs;
plot(t_trimmed, signal_trimmed);
title('Señal recortada');
xlabel('Tiempo [s]');