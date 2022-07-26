<script>
    $(document).ready(function() {

        //datatables
        table = $('#table').DataTable({
            "processing": true, //Feature control the processing indicator.
            "serverSide": true, //Feature control DataTables' server-side processing mode.
            "order": [], //Initial no order.


            // Load data for the table's content from an Ajax source
            "ajax": {
                "url": "<?= base_url() ?>users/ajax_list/<?= $role_list ?>",
                "type": "POST",
            },

            //Set column definition initialisation properties.
            "columnDefs": [{
                    "targets": [0], //first column
                    "orderable": false, //set not orderable
                },
                {
                    "targets": [-1], //last column
                    "orderable": false, //set not orderable
                },
            ],
        });

        //set input/textarea/select event when change value, remove class error and remove text help block
        $("input").change(function() {
            $(this).parent().parent().removeClass('has-error');
            $(this).next().empty();
        });
        $("textarea").change(function() {
            $(this).parent().parent().removeClass('has-error');
            $(this).next().empty();
        });
        $("select").change(function() {
            $(this).parent().parent().removeClass('has-error');
            $(this).next().empty();
        });

        //check all
        $("#check-all").click(function() {
            $(".data-check").prop('checked', $(this).prop('checked'));
            showBottomDelete();
        });

    });

    function showBottomDelete() {
        var total = 0;

        $('.data-check').each(function() {
            total += $(this).prop('checked');
        });

        if (total > 0)
            $('#deleteList').show();
        else
            $('#deleteList').hide();
    }

    function add() {
        save_method = 'add';
        $('#form')[0].reset(); // reset form on modals
        $('.form-group').removeClass('has-error'); // clear error class
        $('.help-block').empty(); // clear error string
        $('#plat_nomor').hide();
        $('#modal_form').modal({
            backdrop: 'static',
        });
        $('.modal-title').text('Tambah Data'); // Set Title to Bootstrap modal title
    }

    function addPlat(id) {
        save_method = 'addPlat';
        $('#form')[0].reset(); // reset form on modals
        $('.form-group').removeClass('has-error'); // clear error class
        $('.help-block').empty(); // clear error string
        $('[name="id"]').val(id);
        $('#fullname').hide();
        $('#email').hide();
        $('#password').hide();
        $('#repassword').hide();
        $('#modal_form').modal({
            backdrop: 'static',
        });
        $('.modal-title').text('Tambah Data'); // Set Title to Bootstrap modal title
    }

    function edit(id) {
        save_method = 'update';
        $('#form')[0].reset(); // reset form on modals
        $('.form-group').removeClass('has-error'); // clear error class
        $('.help-block').empty(); // clear error string
        $('#plat_nomor').hide();
        $('#password').hide();
        $('#repassword').hide();
        $('#modal_form').modal({
            backdrop: 'static',
        });

        //Ajax Load data from ajax
        $.ajax({
            url: "<?= base_url() ?>users/ajax_edit/" + id,
            type: "GET",
            dataType: "JSON",
            success: function(data) {
                alertify.success('Aksi berhasil');
                $('[name="id"]').val(data.id);
                $('[name="fullname"]').val(data.fullname);
                $('[name="email"]').val(data.email);
                $('#modal_form').modal('show'); // show bootstrap modal when complete loaded
                $('.modal-title').text('Ubah Data'); // Set title to Bootstrap modal title

            },
            error: function(jqXHR, textStatus, errorThrown) {
                alert('Error getting data from ajax');
            }
        });
    }

    function reloadTable() {
        table.ajax.reload(null, false); //reload datatable ajax
        $('#deleteList').hide();
    }

    function save() {
        $('#btnSave').text('Menyimpan...'); //change button text
        $('#btnSave').attr('disabled', true); //set button disable
        var url;

        if (save_method == 'add') {
            url = "<?= base_url() ?>users/ajax_add/<?= $role_list ?>";
        }
        if (save_method == 'update') {
            url = "<?= base_url() ?>users/ajax_update";
        }
        if (save_method == 'addPlat') {
            url = "<?= base_url() ?>plats/ajax_add";
        }

        // ajax adding data to database
        $.ajax({
            url: url,
            type: "POST",
            data: $('#form').serialize(),
            dataType: "JSON",
            success: function(data) {
                alertify.success('Berhasil menambahkan data');
                if (data.status) //if success close modal and reload ajax table
                {
                    $("#modal_form").load(location.href + " #modal_form>*", "");
                    $('#modal_form').modal('hide');
                    reloadTable();
                } else {
                    for (var i = 0; i < data.inputerror.length; i++) {
                        alertify.error(data.error_string[i]);
                        $('[name="' + data.inputerror[i] + '"]').parent().parent().addClass('has-error'); //select parent twice to select div form-group class and add has-error class
                        $('[name="' + data.inputerror[i] + '"]').next().text(data.error_string[i]); //select span help-block class set text error string
                    }
                }
                $(".modal-footer").load(location.href + " .modal-footer>*", "");


            },
            error: function(jqXHR, textStatus, errorThrown) {
                alertify.error('Gagal menambahkan atau merubah data, silakan cek kembali masukan anda');
                $('#btnSave').text('Save'); //change button text
                $('#btnSave').attr('disabled', false); //set button enable

            }
        });
    }

    function deleteData(id) {
        if (confirm('Apakah anda yakin akan menghapus data ini?')) {
            // ajax delete data to database
            $.ajax({
                url: "<?= base_url() ?>users/ajax_delete/" + id,
                type: "POST",
                dataType: "JSON",
                success: function(data) {
                    alertify.success('Data berhasil dihapus');
                    //if success reload ajax table
                    $('#modal_form').modal('hide');
                    reloadTable();
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    alertify.error('Gagal menghapus data');
                    alert('Gagal menghapus data, error server silakan hubingi developer');
                }
            });

        }
    }

    function resetPassword(id) {
        if (confirm('Password akan direset menjadi 1234, apakah anda yakin?')) {
            // ajax password data to database
            $.ajax({
                url: "<?= base_url() ?>users/ajax_password/" + id,
                type: "POST",
                dataType: "JSON",
                success: function(data) {
                    alertify.success('Password berhasil diubah');
                    //if success reload ajax table
                    $('#modal_form').modal('hide');
                    reloadTable();
                },
                error: function(jqXHR, textStatus, errorThrown) {
                    alertify.error('Gagal reset password');
                    alert('Gagal reset password, error server silakan hubingi developer');
                }
            });

        }
    }

    function deleteList() {
        var list_id = [];
        $(".data-check:checked").each(function() {
            list_id.push(this.value);
        });
        if (list_id.length > 0) {
            if (confirm('Apakah anda yakin akan menghapus ' + list_id.length + ' data?')) {
                $.ajax({
                    type: "POST",
                    data: {
                        id: list_id
                    },
                    url: "<?= base_url() ?>users/ajax_list_delete",
                    dataType: "JSON",
                    success: function(data) {
                        if (data.status) {
                            alertify.success(+list_id.length + ' Data berhasil dihapus');
                            reloadTable();
                        } else {
                            alertify.error('Gagal menghapus data');
                            alert('Failed.');
                        }

                    },
                    error: function(jqXHR, textStatus, errorThrown) {
                        alertify.error('Gagal menghapus data, error server silakan hubingi developer');
                    }
                });
            }
        } else {
            alert('Tidak ada data dipilih');
        }
    }

    function close_modal() {
        $('#modal_form').modal('hide');
        $("#modal_form").load(location.href + " #modal_form>*", "");
    }
</script>