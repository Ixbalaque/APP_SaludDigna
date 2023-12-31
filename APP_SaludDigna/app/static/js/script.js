document.addEventListener("DOMContentLoaded", function() {
    var enlaces = document.querySelectorAll(".barra-navegacion a");

    enlaces.forEach(function(enlace, index) {
        enlace.addEventListener("click", function() {
            // Remover la clase "selected" de todos los enlaces
            enlaces.forEach(function(e) {
                e.classList.remove("selected");
            });

            // Agregar la clase "selected" al enlace clicado
            this.classList.add("selected");

            // Llamar a la función mostrarContenido con el texto del enlace clicado
            mostrarContenido(enlace.textContent.toLowerCase());
        });
    });

    // Establecer el estilo inicial
    enlaces[0].classList.add("selected");
    mostrarContenido(enlaces[0].textContent.toLowerCase());
});


document.getElementById('formModDataUser').addEventListener('submit', function(event) {
    // Realiza cualquier validación necesaria aquí
    if (!enviarEncuesta()) {
        event.preventDefault();  // Evitar el envío del formulario si enviarEncuesta() devuelve false
    }
});


function mostrarContenido(seccion) {
    console.log('Clic en ' + seccion);

    // Obtener los elementos por su ID
    var encuestaForm = document.getElementById('encuestaForm');
    var modDataUser = document.getElementById('modDataUser');
    var DataContacto = document.getElementById('DataContacto');
    var Mensaje_Res  = document.getElementById('Mensaje_Res');

    // Ocultar todos los elementos
    Mensaje_Res.style.display  = 'none';    
    encuestaForm.style.display = 'none';
    modDataUser.style.display = 'none';
    DataContacto.style.display = 'none';

    // Mostrar el elemento correspondiente a la sección seleccionada
    if (seccion === 'test') {
        console.log('Mostrar Form');
        encuestaForm.style.display = 'block';
    } else if (seccion === 'cuenta') {
        console.log('Mostrar ModUser');
        modDataUser.style.display = 'block';
    } else if (seccion === 'contacto') {
        console.log('Mostrar Contacto');
        DataContacto.style.display = 'block';
    } else {
        console.log('Ocultar todo');
    }
}

function Actualizar_DataUser( key ) {
    
    /*console.log(key);*/
    
    var form = document.getElementById('formModDataUser');
    
    // Crear un nuevo elemento de tipo input (campo oculto)
    var keyInput = document.createElement('input');
    keyInput.type = 'hidden';  // Tipo de input oculto
    keyInput.name = 'key_R';   // Nombre del campo, debe coincidir con el nombre en el servidor
    keyInput.id   = 'key_E';   
    keyInput.value = key;      // Valor que deseas enviar, en este caso, la clave
    
    
    // Agregar el campo oculto al formulario
    
    var formData = new FormData(form);

    // Agregar la clave al FormData
    formData.append('key_R', key);

    
    fetch('/enviar_NewDataUser', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Actualiza el contenido del elemento con el mensaje recibido
        eval(data.js_code);
        document.getElementById('Mensaje_SolicitudNewUser').innerHTML = data.mensaje;
       
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
 
function enviarEncuesta()
{
    var encuestaForm = document.getElementById('encuestaForm');
    var Mensaje_Res  = document.getElementById('Mensaje_Res');
    encuestaForm.style.display = 'none';  
    Mensaje_Res.style.display  = 'block';    
    return false; 
} 

function Aceptar_Resultado()
{
  // Ocultar el mensaje de resultado
  var Mensaje_Res = document.getElementById('Mensaje_Res');
 
  // Mostrar el formulario de encuesta
  var encuestaForm = document.getElementById('encuestaForm');
 
}