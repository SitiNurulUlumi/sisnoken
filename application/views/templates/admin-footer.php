<!-- /.content-wrapper -->
<footer class="main-footer">
  <div class="container">
    <div class="pull-right hidden-xs">
      <b>Versi</b> Beta | Dimuat dalam {elapsed_time} detik.
    </div>
    <strong>Copyright &copy; <script>
        new Date().getFullYear() > 2017 && document.write(new Date().getFullYear());
      </script> SISNOKEN APP.</strong> All rights
    reserved.
  </div>
  <!-- /.container -->
</footer>
</div>
<!-- ./wrapper -->

<!-- jQuery 3 -->
<script src="<?= base_url() ?>assets/bower_components/jquery/dist/jquery.min.js"></script>
<!-- Bootstrap 3.3.7 -->
<script src="<?= base_url() ?>assets/bower_components/bootstrap/dist/js/bootstrap.min.js"></script>
<!-- SlimScroll -->
<script src="<?= base_url() ?>assets/bower_components/jquery-slimscroll/jquery.slimscroll.min.js"></script>
<!-- FastClick -->
<script src="<?= base_url() ?>assets/bower_components/fastclick/lib/fastclick.js"></script>
<!-- AdminLTE App -->
<script src="<?= base_url() ?>assets/dist/js/adminlte.min.js"></script>
<!-- AdminLTE for demo purposes -->
<script src="<?= base_url() ?>assets/dist/js/demo.js"></script>
<!-- Datatables -->
<script src="<?= base_url() ?>assets/bower_components/datatables/js/jquery.dataTables.min.js"></script>
<script src="<?= base_url() ?>assets/bower_components/datatables/js/dataTables.bootstrap.js"></script>
<!-- PACE -->
<script src="<?= base_url() ?>assets/bower_components/PACE/pace.min.js"></script>
<!-- iCheck 1.0.1 -->
<!-- <script src="<?= base_url() ?>assets/plugins/iCheck/icheck.min.js"></script> -->
<script>
  $(document).ajaxStart(function() {
    Pace.restart();
  });

  var delay = alertify.get('notifier', 'delay');
  alertify.set('notifier', 'delay', 4);
  alertify.set('notifier', 'position', 'top-right');
</script>

</body>

</html>