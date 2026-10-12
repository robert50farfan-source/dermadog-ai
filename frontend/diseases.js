// Disease information cards, keyed by the model class names (class_to_idx.json).
// contagious: 'si' | 'no' | 'depende'
const DISEASE_INFO = {
    Dermatitis: {
        description: 'Inflamación de la piel. Puede deberse a alergias, irritantes, parásitos o infecciones, por lo que es un signo que requiere identificar la causa de fondo.',
        signs: ['Enrojecimiento', 'Picor y rascado frecuente', 'Descamación o costras', 'Pérdida de pelo en la zona afectada'],
        contagious: 'depende',
        contagiousText: 'Depende de la causa. La dermatitis por alergia o irritación no se contagia; si hay parásitos o infección, sí puede transmitirse.',
        recommendation: 'Consulte al veterinario para determinar el origen. Evite que el perro se rasque o lama la zona y no aplique cremas de uso humano sin indicación.'
    },
    Fungal_infections: {
        description: 'Infección de la piel causada por hongos, con frecuencia levaduras como Malassezia, que proliferan cuando la piel está húmeda, grasa o con las defensas alteradas.',
        signs: ['Piel grasa y con olor rancio', 'Enrojecimiento y picor', 'Piel engrosada u oscurecida', 'Suele aparecer en oídos, pliegues, axilas y entre los dedos'],
        contagious: 'no',
        contagiousText: 'Por lo general no se contagia; Malassezia vive normalmente en la piel y causa problemas cuando se desequilibra.',
        recommendation: 'El veterinario puede confirmarlo con una citología y suele indicar champús o medicamentos antifúngicos. Mantenga la piel seca, sobre todo en los pliegues.'
    },
    Healthy: {
        description: 'No se detectaron signos compatibles con las enfermedades que el modelo conoce.',
        signs: ['Piel sin enrojecimiento ni lesiones visibles', 'Pelaje uniforme'],
        contagious: 'no',
        contagiousText: 'No aplica.',
        recommendation: 'Mantenga revisiones veterinarias periódicas, buena higiene y control antiparasitario. Si nota picor, olor o pérdida de pelo, consulte al veterinario aunque la foto no muestre lesiones.'
    },
    Hypersensitivity: {
        description: 'Reacción alérgica de la piel. Las causas más comunes son la picadura de pulgas, alimentos o alérgenos del ambiente como polen y ácaros del polvo (dermatitis atópica).',
        signs: ['Picor intenso', 'Enrojecimiento', 'Lamido o mordisqueo de patas', 'Otitis que se repiten'],
        contagious: 'no',
        contagiousText: 'No se contagia: es una respuesta del sistema inmune de cada perro.',
        recommendation: 'El veterinario ayudará a identificar el alérgeno (dieta de eliminación, pruebas). Mantenga un control estricto de pulgas en el perro y en casa.'
    },
    demodicosis: {
        description: 'Enfermedad causada por la proliferación del ácaro Demodex, que vive normalmente en los folículos del pelo. Aparece cuando las defensas no logran controlarlo, sobre todo en cachorros o perros con otras enfermedades.',
        signs: ['Pérdida de pelo en parches, a menudo alrededor de ojos, hocico y patas', 'Descamación', 'Piel enrojecida u oscurecida', 'Infecciones secundarias en casos generalizados'],
        contagious: 'no',
        contagiousText: 'No se contagia a otros perros adultos ni a personas. Se transmite de la madre a los cachorros en los primeros días de vida.',
        recommendation: 'El veterinario la confirma con un raspado de piel e indica un tratamiento acaricida. Es importante revisar si hay una enfermedad de base que baje las defensas.'
    },
    ringworm: {
        description: 'Infección por hongos dermatofitos (Microsporum, Trichophyton) que atacan el pelo y la capa superficial de la piel. También se conoce como dermatofitosis.',
        signs: ['Zonas sin pelo, a menudo circulares', 'Descamación y costras', 'Pelo quebradizo en el borde de la lesión', 'El picor puede ser leve o no existir'],
        contagious: 'si',
        contagiousText: 'Sí, es contagiosa para otros animales y para las personas (zoonosis), por contacto directo o con objetos contaminados.',
        recommendation: 'Lávese las manos después de tocar al perro, limpie camas y accesorios y limite el contacto con niños y otras mascotas. El veterinario puede confirmarla con cultivo o lámpara de Wood.'
    }
};
