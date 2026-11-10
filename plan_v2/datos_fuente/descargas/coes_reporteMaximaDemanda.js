var controlador = siteRoot + 'Medidores/maximademanda/'

$(function () {
    $("#cbCentral").val("1");

    toastr.options = {
        "closeButton": true,
        "debug": false,
        "newestOnTop": false,
        "progressBar": true,
        "positionClass": "toast-top-right",
        "preventDuplicates": false,
        "onclick": null,
        "showDuration": "3000",
        "hideDuration": "1000",
        "timeOut": "5000",
        "extendedTimeOut": "1000",
        "showEasing": "swing",
        "hideEasing": "linear",
        "showMethod": "fadeIn",
        "hideMethod": "fadeOut"
    }

    $("#cbTipoGen").change(function () {
        listarEmpresaByTipoGeneracion();
    });
    $("#cbCentral").change(function () {
        consultar();
    });
    $("#cbEmpresa").change(function () {
        consultar();
    });

    $('#mes').change({
        format: 'Y-m',
        onSelect: function () {
            consultar();
        }
    });

    $('#btnConsultar').click(function () {
        consultar();
    });

    $('#btnExportar').click(function () {
        exportar();
    });

    consultar();
});

function consultar() {
    $("#reporteMaxDemanda").html("");
    validarConsulta();
}

function listarEmpresaByTipoGeneracion() {
    var tipoGeneracion = $('#cbTipoGen').val();

    $('#cbEmpresa').empty();
    $('#cbEmpresa').append('<option value="-1">TODOS</option>');

    $.ajax({
        type: 'POST',
        url: controlador + 'ListarEmpresaXTipoGeneracion',
        data: {
            tipoGeneracion: tipoGeneracion
        },
        success: function (result) {
            if (result.length > 0) {
                for (var i = 0; i < result.length; i++) {
                    $('#cbEmpresa').append('<option value="' + result[i].Emprcodi + '">' + result[i].Emprnomb + '</option>');
                }
            }
            consultar();
        },
        error: function () {
            alert('Ha ocurrido un error.');
        }
    });
}

function validarConsulta() {
    var idParametro = $("#parametro").val();
    var mes = $("#mes").val();

    $.ajax({
        type: 'POST',
        async: false,
        url: controlador + 'GetRangoAnalisis',
        data: {
            idParametro: idParametro,
            mes: mes
        },
        success: function (result) {
            if (result == "true") {
                mostrarResumenMaximaDemanda();
            } else {
                toastr.warning('No se registro Horas Punta para Potencia Activa para el mes indicado');
            }
        }
    });
}

function mostrarResumenMaximaDemanda() {
    var idEmpresa = $("#cbEmpresa").val();
    var mes = $("#mes").val();
    var tipoGeneracion = $('#cbTipoGen').val();
    var tipoCentral = $('#cbCentral').val();

    $.ajax({
        type: 'POST',
        url: controlador + 'ReporteMaximaDemanda',
        data: {
            tipoCentral: tipoCentral,
            tipoGeneracion: tipoGeneracion,
            idEmpresa: idEmpresa,
            mes: mes
        },
        success: function (data) {
            $('#reporteMaxDemanda').html(data);

            $('#tablaConsolidado').dataTable({
                "scrollY": 430,
                "scrollX": true,
                "sDom": 't',
                "ordering": false,
                "bPaginate": false,
                "iDisplayLength": -1
            });
        },
        error: function () {
            alert("Ha ocurrido un error");
        }
    });
}

function exportar() {
    var idEmpresa = $("#cbEmpresa").val();
    var mes = $("#mes").val();
    var tipoGeneracion = $('#cbTipoGen').val();
    var tipoCentral = $('#cbCentral').val();

    $.ajax({
        type: 'POST',
        url: controlador + 'ExportarMaximaDemanda',
        data: {
            tipoCentral: tipoCentral,
            tipoGeneracion: tipoGeneracion,
            idEmpresa: idEmpresa,
            mes: mes
        },
        success: function (result) {
            if (result.length > 0) {
                archivo = result[0];
                nombre = result[1];
                if (archivo != '-1') {
                    window.location.href = controlador + 'DescargarArchivo?rutaArchivoTemp=' + archivo + "&nombreArchivo=" + nombre;
                } else {
                    alert("Error en descargar el archivo");
                }
            }
            else {
                alert("Error en descargar el archivo");
            }
        },
        error: function () {
            alert('ha ocurrido un error al descargar el archivo excel.');
        }
    });
}