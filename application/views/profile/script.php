<script type="text/javascript">
    function save() {
        $('#btnSave').text('Menyimpan...'); //change button text
        $('#btnSave').attr('disabled', true); //set button disable
        var url;


        url = "<?= base_url() ?>profile/ajax_save";

        // ajax adding data to database
        $.ajax({
            url: url,
            type: "POST",
            data: $('#form').serialize(),
            dataType: "JSON",
            success: function(data) {
                if (data.status) //if success close modal and reload ajax table
                {
                    $("#form").load(location.href + " #form>*", "");
                    alertify.success('Berhasil memperbarui data.');
                    success();
                } else {
                    $("#form-footer").load(location.href + " #form-footer>*", "");
                    for (var i = 0; i < data.inputerror.length; i++) {
                        alertify.error(data.error_string[i]);
                        $('[name="' + data.inputerror[i] + '"]').parent().parent().addClass('has-error'); //select parent twice to select div form-group class and add has-error class
                        $('[name="' + data.inputerror[i] + '"]').next().text(data.error_string[i]); //select span help-block class set text error string
                    }
                }
            },
            error: function(jqXHR, textStatus, errorThrown) {
                alertify.error('Gagal menambahkan atau merubah data, silakan cek kembali masukan anda');
                $("#form-footer").load(location.href + " #form-footer>*", "");
            }
        });
    }
</script>