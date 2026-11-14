# Set de oro

Fotos externas al dataset, con su clase confirmada (idealmente por un veterinario).
Copie las fotos en esta carpeta y agregue una fila por foto en manifest.csv:

    archivo,clase_esperada,fuente
    tina_01.jpg,ringworm,foto propia
    sarna_01.jpg,fuera_de_clase,https://...
    gato_01.jpg,no_perro,foto propia

clase_esperada: Dermatitis, Fungal_infections, Healthy, Hypersensitivity, demodicosis, ringworm,
fuera_de_clase (enfermedad que el modelo no conoce, por ejemplo sarna sarcóptica) o no_perro.

Empiece por las fotos que ya fallaron: las 8 de tiña y los casos de sarna sarcóptica.
